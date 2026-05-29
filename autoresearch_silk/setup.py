#!/usr/bin/env python
"""ONE-TIME setup for the silk autoresearch loop. Run this before experimenting.

Downloads `lamm-mit/silkome-full` (private — needs `huggingface-cli login`), keeps
sequence + the 4 mechanical targets, removes train/test sequence overlap, and CACHES ESMC
embeddings (mean-pooled + per-residue, fp16) so experiments are fast and fully local.

Backbone is configurable — caches are per-model so several can coexist:
  python setup.py                              # ESMC-300M (default; also reads config.json)
  python setup.py --model biohub/ESMC-600M
  python setup.py --model biohub/ESMC-6B --device cuda     # e.g. on a DGX Spark

Outputs (gitignored except the small parquets):
  data/{train,test}.parquet                    # sequence + idv + category1 + 4 targets
  cache/<slug>_{train,test}_mean.npy           # (N, d) mean-pooled embeddings
  cache/<slug>_{train,test}_resid.npz          # flat per-residue fp16 + per-seq lengths
  cache/<slug>_meta.json                       # {model, dim}
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


def slug(model):
    return model.split("/")[-1]


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


def load_silkome():
    from datasets import load_dataset
    tr = load_dataset("lamm-mit/silkome-full", split="train").to_pandas()
    te = load_dataset("lamm-mit/silkome-full", split="test").to_pandas()
    tr = tr.dropna(subset=TARGETS)[KEEP].reset_index(drop=True)
    te = te.dropna(subset=TARGETS)[KEEP].reset_index(drop=True)
    overlap = set(tr["sequence"]) & set(te["sequence"])
    if overlap:                            # avoid train/test leakage
        tr = tr[~tr["sequence"].isin(overlap)].reset_index(drop=True)
        print(f"[setup] dropped {len(overlap)} train rows whose sequence is in test")
    return tr, te


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
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--data-only", action="store_true",
                    help="regenerate data/*.parquet only (e.g. to add meta columns); skip embedding. "
                         "Row order is deterministic, so existing caches stay aligned.")
    args = ap.parse_args()

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    sl = slug(args.model)

    tr, te = load_silkome()
    tr.to_parquet(os.path.join(HERE, "data", "train.parquet"))
    te.to_parquet(os.path.join(HERE, "data", "test.parquet"))
    print(f"[setup] train={len(tr)} test={len(te)} rows saved to data/ "
          f"(cols: {', '.join(c for c in tr.columns if c != 'sequence')})")
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
        np.save(os.path.join(HERE, "cache", f"{sl}_{name}_mean.npy"), mean)
        np.savez(os.path.join(HERE, "cache", f"{sl}_{name}_resid.npz"),
                 resid=resid, lengths=lengths)
        print(f"[setup] {name}: mean{mean.shape} resid{resid.shape} ({time.time()-t0:.0f}s)")

    json.dump({"model": args.model, "dim": int(dim)},
              open(os.path.join(HERE, "cache", f"{sl}_meta.json"), "w"))
    print(f"[setup] done. Cached backbone '{args.model}' (dim={dim}). "
          f"Set this model in config.json to use it in experiments.")


if __name__ == "__main__":
    main()
