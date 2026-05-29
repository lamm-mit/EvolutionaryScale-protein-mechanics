#!/usr/bin/env python
"""esm_embed — compute ESMC embeddings for sequence(s) and save them.

Examples
--------
  python esm_embed.py --seq GGAGQGGYGGLGSQ... --out emb            # -> emb.npy (1 x d) + emb.csv
  python esm_embed.py --fasta proteins.fasta --pool mean --out emb # mean-pooled, one row per record
  python esm_embed.py --seq MQIF... --pool none --out perres       # per-residue: perres_query.npy (L x d)

Mean pooling gives one vector per protein (good for comparison/training). Per-residue gives (L, d).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esm_common import collect_inputs, load_esmc, embed_sequences, log
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Compute ESMC embeddings -> .npy / .csv")
    ap.add_argument("--seq")
    ap.add_argument("--fasta")
    ap.add_argument("--model", default="biohub/ESMC-300M")
    ap.add_argument("--pool", choices=["mean", "none"], default="mean")
    ap.add_argument("--out", default="embeddings", help="output prefix")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    inputs = collect_inputs(args.seq, args.fasta)
    names, seqs = list(inputs), list(inputs.values())
    tok, model, device = load_esmc(args.model, None if args.device == "auto" else args.device)
    log(f"[esm_embed] {len(seqs)} seq(s), model={args.model}, device={device}, pool={args.pool}")

    embs = embed_sequences(tok, model, seqs, device, pool=args.pool, batch_size=args.batch_size)

    if args.pool == "mean":
        mat = np.stack(embs)                                  # (N, d)
        np.save(args.out + ".npy", mat)
        with open(args.out + ".csv", "w") as fh:
            fh.write("name," + ",".join(f"d{i}" for i in range(mat.shape[1])) + "\n")
            for n, v in zip(names, mat):
                fh.write(n + "," + ",".join(f"{x:.5f}" for x in v) + "\n")
        print(f"wrote {args.out}.npy {mat.shape} and {args.out}.csv")
    else:
        for n, v in zip(names, embs):
            np.save(f"{args.out}_{n}.npy", v)
            log(f"  {n}: {v.shape}")
        print(f"wrote {len(embs)} per-residue arrays as {args.out}_<name>.npy")


if __name__ == "__main__":
    main()
