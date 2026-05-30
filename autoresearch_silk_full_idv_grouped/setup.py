#!/usr/bin/env python
"""ONE-TIME setup for the grouped silk autoresearch problem.

Downloads an idv-grouped silkome dataset, keeps one row per fiber/property id, and caches ESMC
embeddings for each individual sequence in each row. This deliberately does NOT concatenate the
proteins before ESMC. Each sequence is tokenized and embedded independently, then later aggregated by
the model at the idv/sample level.

Default:
  python setup.py

Outputs:
  data/<dataset>_{train,test}.parquet
  cache/<dataset>__<model>_{train,test}_seq_mean.npy      # (N, Smax, d) float32
  cache/<dataset>__<model>_{train,test}_seq_mask.npy      # (N, Smax) bool
  cache/<dataset>__<model>_{train,test}_category_ids.npy  # (N, Smax) int64
  cache/<dataset>__<model>_{train,test}_seq_lengths.npy   # (N, Smax) int32
  cache/<dataset>__<model>_meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Iterable

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_TARGETS = ["toughness", "E", "strength", "strain"]
NORM_TARGETS = ["toughnessNorm", "ENorm", "strengthNorm", "strainNorm"]


def dataset_slug(dataset: str) -> str:
    return dataset.split("/")[-1]


def cache_key(dataset: str, model: str) -> str:
    return f"{dataset_slug(dataset)}__{model.split('/')[-1]}"


def device_for(model: str, prefer: str) -> str:
    import torch

    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if "6B" in model:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def target_columns(target_mode: str) -> list[str]:
    if target_mode == "raw":
        return RAW_TARGETS
    if target_mode == "norm":
        return NORM_TARGETS
    raise ValueError(f"Unsupported target_mode: {target_mode}")


def _load_split(dataset: str, split: str, targets: list[str]):
    from datasets import load_dataset

    df = load_dataset(dataset, split=split).to_pandas()
    required = ["idv", "sequences", "sequence_categories", "property_tuple_key"] + targets
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"{dataset} split={split} missing required columns: {missing}")
    df = df.dropna(subset=targets).reset_index(drop=True)
    return df


def load_grouped_silkome(dataset: str, target_mode: str):
    targets = target_columns(target_mode)
    train = _load_split(dataset, "train", targets)
    test = _load_split(dataset, "test", targets)
    overlap = set(train["idv"]) & set(test["idv"])
    if overlap:
        raise SystemExit(f"Train/test idv leakage: {len(overlap)} overlapping idv values")
    return train, test, targets


def category_classes(*frames) -> list[str]:
    cats = set()
    for df in frames:
        for row in df["sequence_categories"]:
            cats.update(str(c) for c in row)
    return sorted(cats)


def flatten_unique_sequences(*frames) -> list[str]:
    seen = {}
    for df in frames:
        for row in df["sequences"]:
            for seq in row:
                s = str(seq)
                if s not in seen:
                    seen[s] = None
    return list(seen)


def embed_sequences(sequences: list[str], tok, model, device: str, batch_size: int) -> dict[str, np.ndarray]:
    """Return sequence -> mean ESMC residue embedding."""
    import torch

    out: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for i in range(0, len(sequences), batch_size):
            chunk = sequences[i : i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True).to(device)
            h = model(**enc, output_hidden_states=True).hidden_states[-1]
            am = enc["attention_mask"]
            for r, seq in enumerate(chunk):
                L = int(am[r].sum())
                # ESMC tokenizer adds boundary tokens; keep only amino-acid residue states.
                res = h[r, 1 : L - 1].float().cpu()
                out[seq] = res.mean(0).numpy().astype("float32")
            if (i // batch_size) % 20 == 0:
                print(f"  embedded {min(i + batch_size, len(sequences))}/{len(sequences)}", flush=True)
    return out


def pack_split(df, emb: dict[str, np.ndarray], cat_to_id: dict[str, int]):
    n = len(df)
    smax = int(df["n_sequences"].max())
    dim = len(next(iter(emb.values())))
    X = np.zeros((n, smax, dim), dtype="float32")
    mask = np.zeros((n, smax), dtype=bool)
    cat_ids = np.zeros((n, smax), dtype="int64")
    lengths = np.zeros((n, smax), dtype="int32")

    for i, row in df.iterrows():
        seqs = [str(s) for s in row["sequences"]]
        cats = [str(c) for c in row["sequence_categories"]]
        for j, (seq, cat) in enumerate(zip(seqs, cats)):
            X[i, j] = emb[seq]
            mask[i, j] = True
            cat_ids[i, j] = cat_to_id[cat]
            lengths[i, j] = len(seq)
    return X, mask, cat_ids, lengths


def save_split(name: str, df, targets: list[str], key: str, emb, cat_to_id: dict[str, int]):
    dsl = key.split("__")[0]
    df.to_parquet(os.path.join(HERE, "data", f"{dsl}_{name}.parquet"))
    X, mask, cat_ids, lengths = pack_split(df, emb, cat_to_id)
    np.save(os.path.join(HERE, "cache", f"{key}_{name}_seq_mean.npy"), X)
    np.save(os.path.join(HERE, "cache", f"{key}_{name}_seq_mask.npy"), mask)
    np.save(os.path.join(HERE, "cache", f"{key}_{name}_category_ids.npy"), cat_ids)
    np.save(os.path.join(HERE, "cache", f"{key}_{name}_seq_lengths.npy"), lengths)
    print(f"[setup] {name}: groups={len(df)} X{X.shape} targets={targets}")
    return int(X.shape[-1])


def main():
    ap = argparse.ArgumentParser(description="Cache grouped silkome data + per-sequence ESMC embeddings.")
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    ap.add_argument("--dataset", default=cfg.get("dataset", "lamm-mit/silkome-full-idv-grouped"))
    ap.add_argument("--model", default=cfg.get("esmc_model", "biohub/ESMC-300M"))
    ap.add_argument("--target-mode", default=cfg.get("target_mode", "raw"), choices=["raw", "norm"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--data-only", action="store_true", help="write data/*.parquet and metadata only")
    ap.add_argument("--smoke-test", action="store_true", help="embed a few sequences and print shapes only")
    args = ap.parse_args()

    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "cache"), exist_ok=True)
    key = cache_key(args.dataset, args.model)

    train, test, targets = load_grouped_silkome(args.dataset, args.target_mode)
    cats = category_classes(train, test)
    cat_to_id = {c: i + 1 for i, c in enumerate(cats)}  # 0 is padding
    seqs = flatten_unique_sequences(train, test)
    print(
        f"[setup] dataset={args.dataset} target_mode={args.target_mode} "
        f"train={len(train)} test={len(test)} unique_sequences={len(seqs)} categories={len(cats)}"
    )
    print(
        f"[setup] max sequences/group={max(train['n_sequences'].max(), test['n_sequences'].max())} "
        f"max sequence length={max(len(s) for s in seqs)}"
    )

    # Always write parquets, even in smoke/data-only mode.
    dsl = dataset_slug(args.dataset)
    train.to_parquet(os.path.join(HERE, "data", f"{dsl}_train.parquet"))
    test.to_parquet(os.path.join(HERE, "data", f"{dsl}_test.parquet"))

    if args.data_only:
        json.dump(
            {
                "model": args.model,
                "dataset": args.dataset,
                "target_mode": args.target_mode,
                "targets": targets,
                "category_classes": cats,
                "data_only": True,
            },
            open(os.path.join(HERE, "cache", f"{key}_meta.json"), "w"),
            indent=2,
        )
        print("[setup] --data-only: wrote grouped parquets; embeddings skipped.")
        return

    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = device_for(args.model, args.device)
    print(f"[setup] embedding with {args.model} on {device}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model, device_map="auto" if "6B" in args.model and device == "cuda" else None)
    if not ("6B" in args.model and device == "cuda"):
        model = (model.float() if device != "cuda" and "6B" in args.model else model).to(device)
    model.eval()

    if args.smoke_test:
        emb = embed_sequences(seqs[: min(8, len(seqs))], tok, model, device, batch_size=min(args.batch_size, 4))
        print(f"[smoke] OK embedded {len(emb)} sequences dim={len(next(iter(emb.values())))}")
        return

    t0 = time.time()
    emb = embed_sequences(seqs, tok, model, device, args.batch_size)
    dim = save_split("train", train, targets, key, emb, cat_to_id)
    save_split("test", test, targets, key, emb, cat_to_id)
    json.dump(
        {
            "model": args.model,
            "dataset": args.dataset,
            "target_mode": args.target_mode,
            "targets": targets,
            "dim": int(dim),
            "category_classes": cats,
            "category_padding_id": 0,
            "train_groups": int(len(train)),
            "test_groups": int(len(test)),
            "unique_sequences": int(len(seqs)),
            "seconds": round(time.time() - t0, 1),
        },
        open(os.path.join(HERE, "cache", f"{key}_meta.json"), "w"),
        indent=2,
    )
    print(f"[setup] done. Cached {len(seqs)} unique sequence embeddings in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
