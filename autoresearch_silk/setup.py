#!/usr/bin/env python
"""ONE-TIME setup for the silk autoresearch loop. Run this before experimenting.

Downloads a silkome dataset (private — needs `huggingface-cli login`), keeps sequence + the 4
mechanical targets + taxonomy, and CACHES ESMC embeddings (mean-pooled + per-residue, fp16) so
experiments are fast and fully local. By default, datasets with provided train/test splits use those
splits; split-less datasets are pooled, deduped by sequence, and re-split into a deterministic
**leakage-safe grouped** train/test (test_frac). Exact-duplicate sequences are dropped in BOTH cases
(and any test sequence also in train), so the cached silkome-masp split is 891/137, not the raw 895/138.

Dataset and backbone are configurable; caches are keyed by (dataset, model) so several coexist:
  python setup.py                                    # config.json defaults (silkome-masp + ESMC-300M)
  python setup.py --dataset lamm-mit/silkome-full    # larger set (3.5k seqs)
  python setup.py --model biohub/ESMC-600M
  python setup.py --model biohub/ESMC-6B --device cuda            # e.g. on a DGX Spark
  python setup.py --smoke-test                       # quick check, no cache written

Outputs (all gitignored — silkome is private):
  data/<dataset>_{train,test}.parquet                # sequence + idv + taxonomy + 4 targets
  cache/<dataset>__<model>_{train,test}_mean.npy     # (N, d) mean-pooled embeddings
  cache/<dataset>__<model>_{train,test}_resid.npz    # flat per-residue fp16 + per-seq lengths
  cache/<dataset>__<model>_meta.json                 # {model, dataset, dim, test_frac, seed}
"""
import argparse, json, os, sys, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TARGETS = ["toughness", "E", "strength", "strain"]
# taxonomy/meta columns kept for fusion / multi-task / auxiliary-classifier experiments
META = ["family", "genus", "species", "category1", "category2", "sex", "ncbi"]
KEEP = ["idv", "sequence"] + META + TARGETS


def dataset_slug(dataset):
    return dataset.split("/")[-1]


def cache_key(dataset, model):
    # cache/data are keyed by dataset+model so several can coexist
    return f"{dataset_slug(dataset)}__{model.split('/')[-1]}"


def device_for(model, prefer):
    import torch
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if "6B" in model:
        return "cpu"                      # MPS lacks ops for the 6B path
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _clean(df):
    return df.dropna(subset=TARGETS)[KEEP].drop_duplicates(subset="sequence").reset_index(drop=True)


def load_silkome(dataset, test_frac=0.15, seed=0, split_mode="auto"):
    """Return (train_df, test_df, mode_used). Drops rows missing any target + dedups by sequence.

    split_mode:
      "provided" — use the dataset's own `train`/`test` splits (drops train seqs that also appear in
                   test to avoid exact-sequence leakage).
      "grouped"  — pool all splits, dedup, and make a deterministic **leakage-safe** split grouped by
                   measured-property tuple (test_frac, seed) so identical-fiber sequences never straddle it.
      "auto"     — use "provided" if the dataset has both train & test splits, else "grouped".
    Works for any silkome-style dataset (e.g. silkome-masp now ships train/test; split-less ones use grouped)."""
    import hashlib
    import pandas as pd
    from datasets import get_dataset_split_names, load_dataset
    avail = get_dataset_split_names(dataset)
    has_provided = "train" in avail and "test" in avail
    use_provided = (split_mode == "provided") or (split_mode == "auto" and has_provided)

    if use_provided:
        if not has_provided:
            raise SystemExit(f"split_mode='provided' but {dataset} lacks train+test splits (has {avail})")
        tr = _clean(load_dataset(dataset, split="train").to_pandas())
        te = _clean(load_dataset(dataset, split="test").to_pandas())
        overlap = set(tr["sequence"]) & set(te["sequence"])
        if overlap:
            tr = tr[~tr["sequence"].isin(overlap)].reset_index(drop=True)
        return tr, te, "provided"

    # grouped: pool all non-reserved splits, dedup, split by property tuple
    splits = [s for s in avail if s != "all"]                 # 'all' = reserved union keyword
    df = _clean(pd.concat([load_dataset(dataset, split=s).to_pandas() for s in splits], ignore_index=True))
    keys = [hashlib.md5(np.round(r, 5).tobytes()).hexdigest() for r in df[TARGETS].to_numpy()]
    uniq = list(dict.fromkeys(keys))
    np.random.default_rng(seed).shuffle(uniq)
    test_groups = set(uniq[:max(1, int(round(test_frac * len(uniq))))])
    is_test = np.array([k in test_groups for k in keys])
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True), "grouped"


def embed_split(df, tok, model, device, batch_size=8):
    """Return (mean (N,d) fp32, flat_resid (sumL,d) fp16, lengths (N,) int32)."""
    import torch
    seqs = df["sequence"].astype(str).tolist()
    means, resid_chunks, lengths = [], [], []
    with torch.inference_mode():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True).to(device)
            h = model(**enc, output_hidden_states=True).hidden_states[-1]   # (B,L,d)
            am = enc["attention_mask"]
            for r in range(len(chunk)):
                L = int(am[r].sum())                # incl. <cls>,<eos>
                res = h[r, 1:L - 1].float().cpu()    # residues only -> (len, d)
                means.append(res.mean(0).numpy())
                resid_chunks.append(res.half().numpy())
                lengths.append(res.shape[0])
            if (i // batch_size) % 20 == 0:
                print(f"  embedded {min(i + batch_size, len(seqs))}/{len(seqs)}", flush=True)
    return (np.stack(means).astype("float32"),
            np.concatenate(resid_chunks).astype("float16"),
            np.asarray(lengths, dtype="int32"))


def main():
    ap = argparse.ArgumentParser(description="Cache silkome data + ESMC embeddings.")
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    ap.add_argument("--model", default=cfg.get("esmc_model", "biohub/ESMC-300M"))
    ap.add_argument("--dataset", default=cfg.get("dataset", "lamm-mit/silkome-masp"),
                    help="HF dataset repo (default lamm-mit/silkome-masp; or lamm-mit/silkome-full)")
    ap.add_argument("--test-frac", type=float, default=cfg.get("test_frac", 0.15))
    ap.add_argument("--split-mode", default=cfg.get("split_mode", "auto"),
                    choices=["auto", "provided", "grouped"],
                    help="auto: use dataset's train/test if present else grouped; "
                         "provided: dataset's own split; grouped: harness leakage-safe split")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--data-only", action="store_true",
                    help="regenerate data/*.parquet only (e.g. to add meta columns); skip embedding. "
                         "Row order is deterministic, so existing caches stay aligned.")
    ap.add_argument("--smoke-test", action="store_true",
                    help="quick end-to-end check: load silkome + embed 4 sequences + print shapes; "
                         "no parquet/cache written. Run this before the full setup.")
    args = ap.parse_args()

    seed = cfg.get("seed", 0)
    if args.smoke_test:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        tr, te, mode = load_silkome(args.dataset, args.test_frac, seed, args.split_mode)
        device = device_for(args.model, args.device)
        tok = AutoTokenizer.from_pretrained(args.model)
        m = AutoModelForMaskedLM.from_pretrained(args.model)
        m = (m.float() if device != "cuda" and "6B" in args.model else m).to(device).eval()
        mean, resid, lengths = embed_split(tr.head(4).reset_index(drop=True), tok, m, device, 2)
        print(f"[smoke] OK | dataset={args.dataset} ({mode} split) model={args.model} device={device} | "
              f"train={len(tr)} test={len(te)} | mean{mean.shape} resid{resid.shape} "
              f"lengths={lengths.tolist()}")
        return

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    key = cache_key(args.dataset, args.model)
    dsl = dataset_slug(args.dataset)

    tr, te, mode = load_silkome(args.dataset, args.test_frac, seed, args.split_mode)
    tr.to_parquet(os.path.join(HERE, "data", f"{dsl}_train.parquet"))
    te.to_parquet(os.path.join(HERE, "data", f"{dsl}_test.parquet"))
    split_desc = f"{mode} split" + (f" (test_frac={args.test_frac}, seed={seed})" if mode == "grouped"
                                    else " (dataset's own train/test)")
    print(f"[setup] dataset={args.dataset} | {split_desc} -> train={len(tr)} test={len(te)} | cols: "
          f"{', '.join(c for c in tr.columns if c != 'sequence')}")
    if args.data_only:
        print("[setup] --data-only: parquets regenerated; embeddings left untouched.")
        return

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    device = device_for(args.model, args.device)
    print(f"[setup] embedding with {args.model} on {device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    model = (model.float() if device != "cuda" and "6B" in args.model else model).to(device).eval()

    dim = None
    for name, df in [("train", tr), ("test", te)]:
        t0 = time.time()
        mean, resid, lengths = embed_split(df, tok, model, device, args.batch_size)
        dim = mean.shape[1]
        np.save(os.path.join(HERE, "cache", f"{key}_{name}_mean.npy"), mean)
        np.savez(os.path.join(HERE, "cache", f"{key}_{name}_resid.npz"), resid=resid, lengths=lengths)
        print(f"[setup] {name}: mean{mean.shape} resid{resid.shape} ({time.time()-t0:.0f}s)")

    json.dump({"model": args.model, "dataset": args.dataset, "dim": int(dim),
               "split_mode": mode, "test_frac": args.test_frac, "seed": seed},
              open(os.path.join(HERE, "cache", f"{key}_meta.json"), "w"))
    print(f"[setup] done. Cached '{args.dataset}' + '{args.model}' (dim={dim}). "
          f"Set `dataset`/`esmc_model` in config.json to use this in experiments.")


if __name__ == "__main__":
    main()
