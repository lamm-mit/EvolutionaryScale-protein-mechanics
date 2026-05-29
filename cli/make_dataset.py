#!/usr/bin/env python
"""make_dataset — build the structural-protein family dataset (UniProt + sliding windows)
and optionally push it to the Hugging Face Hub.

This is the dataset behind the transfer-learning section of `ESM_structural_proteins.ipynb`:
sequences from several structural-protein families, sliced into overlapping windows, with a
`family` label. Used by `esm_train_head.py`.

Examples
--------
  python make_dataset.py --per-family 40 --out ../data/structural_protein_families.csv
  python make_dataset.py --push --repo lamm-mit/structural-protein-families
"""
import argparse, csv, os, sys, urllib.parse, urllib.request

FAMILY_QUERIES = {
    "Spidroin": "protein_name:spidroin AND length:[80 TO 700]",
    "Fibroin":  "protein_name:fibroin AND length:[80 TO 700]",
    "Collagen": "protein_name:collagen AND reviewed:true AND length:[150 TO 700]",
    "Elastin":  "protein_name:elastin AND reviewed:true AND length:[100 TO 700]",
    "Resilin":  "protein_name:resilin AND length:[80 TO 700]",
    "Keratin":  "protein_name:keratin AND reviewed:true AND length:[100 TO 700]",
    "Globular": '(protein_name:lysozyme OR protein_name:myoglobin OR '
                'protein_name:"cytochrome c") AND reviewed:true AND length:[80 TO 450]',
}


def fetch_family(query, size):
    url = (f"https://rest.uniprot.org/uniprotkb/search?query={urllib.parse.quote(query)}"
           f"&format=fasta&size={size}")
    req = urllib.request.Request(url, headers={"User-Agent": "esm-class-notebook"})
    txt = urllib.request.urlopen(req, timeout=60).read().decode()
    recs, acc, buf = [], None, []
    for line in txt.splitlines():
        if line.startswith(">"):
            if acc is not None:
                recs.append((acc, "".join(buf)))
            parts = line[1:].split("|")
            acc = parts[1] if len(parts) > 2 else line[1:].split()[0]
            buf = []
        else:
            buf.append(line.strip())
    if acc is not None:
        recs.append((acc, "".join(buf)))
    return [(a, s) for a, s in recs if 0 < len(s) <= 1000]


def windows(seq, w=200, stride=150, max_windows=4):
    if len(seq) <= w:
        return [seq]
    return [seq[i:i + w] for i in range(0, len(seq) - w + 1, stride)][:max_windows] or [seq[:w]]


CARD_HEADER_KEYS = """license: cc-by-4.0
task_categories:
- text-classification
tags:
- protein
- structural-proteins
- silk
- collagen
- elastin
- biomaterials
- esm
"""

CARD_BODY = """# Structural-protein families (sequence windows)

Amino-acid sequence windows from several **structural / biomaterials protein families**
(spider spidroins, silkworm fibroin, collagen, elastin, resilin, keratin) plus **globular**
controls (lysozyme, myoglobin, cytochrome c). Sequences are fetched from **UniProt** and sliced
into overlapping windows; each row carries a `family` label.

Built for the transfer-learning demo in
[`EvolutionaryScale-protein-mechanics`](https://github.com/lamm-mit/EvolutionaryScale-protein-mechanics)
— train a lightweight head on **frozen ESMC embeddings** to classify family (see `cli/esm_train_head.py`).

## Columns
- `id` — `<family>_<accession>_w<window>`
- `family` — class label
- `accession` — source UniProt accession
- `window` — window index within the source protein
- `sequence` — amino-acid window (≤200 aa)

## Provenance
UniProt REST `search` per family (see `cli/make_dataset.py`), windows of length 200 / stride 150,
≤4 windows per protein. Sequences © their respective UniProt entries.
"""


def main():
    ap = argparse.ArgumentParser(description="Build/push the structural-protein family dataset.")
    ap.add_argument("--per-family", type=int, default=40)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--stride", type=int, default=150)
    ap.add_argument("--max-windows", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data",
                                                  "structural_protein_families.csv"))
    ap.add_argument("--push", action="store_true", help="upload to the HF Hub")
    ap.add_argument("--repo", default="lamm-mit/structural-protein-families")
    args = ap.parse_args()

    rows = []
    for fam, q in FAMILY_QUERIES.items():
        recs = fetch_family(q, args.per_family)
        for acc, seq in recs:
            for wi, win in enumerate(windows(seq, args.window, args.stride, args.max_windows)):
                rows.append({"id": f"{fam}_{acc}_w{wi}", "family": fam, "accession": acc,
                             "window": wi, "sequence": win})
        print(f"{fam:10} {len(recs):3d} proteins", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "family", "accession", "window", "sequence"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")

    if args.push:
        # Push via `datasets` so the repo gets proper parquet + dataset_info -> HF viewer works.
        import pandas as pd
        from datasets import Dataset
        from huggingface_hub import HfApi
        Dataset.from_pandas(pd.read_csv(args.out)).push_to_hub(args.repo)
        # remove any legacy raw CSV left at the repo root (keeps the viewer to one clean config)
        try:
            HfApi().delete_file("structural_protein_families.csv", args.repo, repo_type="dataset")
        except Exception:
            pass
        # write a clean, deterministic dataset card: single valid YAML (tags + explicit parquet
        # config so the viewer works) followed by the human-readable body.
        readme = ("---\n" + CARD_HEADER_KEYS +
                  "configs:\n- config_name: default\n  data_files:\n"
                  "  - split: train\n    path: data/train-*\n"
                  "---\n\n" + CARD_BODY)
        HfApi().upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                            repo_id=args.repo, repo_type="dataset")
        print(f"pushed (datasets) to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
