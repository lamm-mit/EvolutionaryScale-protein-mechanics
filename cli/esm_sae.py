#!/usr/bin/env python
"""esm_sae — extract interpretable Sparse-Autoencoder (SAE) features from ESMC-6B (local).

For each sequence, decompose layer-60 embeddings into 16,384 sparse features (top-64 per residue),
rank the strongest features, and (optionally) fetch their auto-generated descriptions from the
keyless ESM Atlas endpoint.

Examples
--------
  python esm_sae.py --seq GGAGQGGYGGLGSQGAGRGGLGGQGAGAAAAAAAA --topk 10 --out sae
  python esm_sae.py --fasta proteins.fasta --rank prevalence --save-matrix

Outputs: <out>_<name>.csv (feature_id, max, prevalence, label, category, summary) and, with
--save-matrix, <out>_<name>.npy (L x 16384 dense activations). Loads ESMC-6B (~25 GB) on CPU fp32.
"""
import argparse, csv, json, os, sys, urllib.request
from functools import lru_cache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esm_common import collect_inputs, pick_device, log
import numpy as np, torch


@lru_cache(maxsize=16384)
def feature_info(idx):
    url = f"https://biohub.ai/esm/protein/api/v1alpha1/features/{int(idx)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "esm-cli"})
        return json.load(urllib.request.urlopen(req, timeout=20))
    except Exception as e:
        return {"label": "(unavailable)", "category": None, "summary": str(e)[:60]}


def main():
    ap = argparse.ArgumentParser(description="Extract ESMC-6B SAE features (local).")
    ap.add_argument("--seq")
    ap.add_argument("--fasta")
    ap.add_argument("--esmc", default="biohub/ESMC-6B")
    ap.add_argument("--sae", default="biohub/ESMC-6B-sae-k64-codebook16384")
    ap.add_argument("--layer", type=int, default=60)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--rank", choices=["max", "prevalence"], default="max")
    ap.add_argument("--describe", action="store_true", default=True,
                    help="fetch feature descriptions (keyless endpoint); on by default")
    ap.add_argument("--no-describe", dest="describe", action="store_false")
    ap.add_argument("--save-matrix", action="store_true", help="also save (L x 16384) activations")
    ap.add_argument("--out", default="sae")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    inputs = collect_inputs(args.seq, args.fasta)
    device = pick_device(args.device, heavy=True)
    log(f"[esm_sae] loading {args.esmc} + SAE layer {args.layer} on {device}")

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.esmc)
    model = AutoModel.from_pretrained(args.esmc)
    model = (model.float() if device != "cuda" else model).to(device).eval()
    sae = AutoModel.from_pretrained(
        args.sae, allow_patterns=["config.json", f"layer_{args.layer}.safetensors"], device=device)
    sae.initialize_layers([args.layer])
    model.add_sae_models([sae.layers[str(args.layer)]])
    key = f"layer{args.layer}"

    for name, seq in inputs.items():
        enc = {k: v.to(device) for k, v in tok(seq, return_tensors="pt").items()}
        with torch.inference_mode():
            out = model(**enc)
        F = out["sae_outputs"][key].to_dense().float().cpu().numpy()[1:-1]   # (L, 16384)
        score = F.max(0) if args.rank == "max" else (F > 0).sum(0)
        top = np.argsort(score)[::-1][:args.topk]
        maxact = F.max(0); prev = (F > 0).sum(0)

        rows = []
        for fid in top:
            info = feature_info(int(fid)) if args.describe else {}
            rows.append({"feature_id": int(fid), "max_activation": round(float(maxact[fid]), 4),
                         "prevalence": int(prev[fid]),
                         "label": info.get("label", ""), "category": info.get("category", ""),
                         "summary": str(info.get("summary", "")).replace("\n", " ")[:200]})
        out_csv = f"{args.out}_{name}.csv"
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["feature_id", "max_activation", "prevalence",
                                               "label", "category", "summary"])
            w.writeheader(); w.writerows(rows)
        log(f"  {name}: L={F.shape[0]} active/res={ (F>0).sum(1).mean():.0f} -> {out_csv}")
        for r in rows[:5]:
            log(f"     f{r['feature_id']:5d} max={r['max_activation']:.2f} | {r['label']} [{r['category']}]")
        if args.save_matrix:
            np.save(f"{args.out}_{name}.npy", F)

    print(f"done: {len(inputs)} sequence(s), top-{args.topk} features each (rank by {args.rank})")


if __name__ == "__main__":
    main()
