#!/usr/bin/env python
"""esm_mutscan — zero-shot deep mutational scan with ESMC (masked-marginal scoring).

For each position in a window, mask it and score every amino acid by the log-likelihood ratio
    LLR(mut) = log P(mut | context) - log P(wt | context)
Positive = model prefers the mutation; strongly negative = disfavored / likely deleterious.

Examples
--------
  python esm_mutscan.py --seq GGAGQGGYGGLGSQGAGRGGLGGQGAGAAAAAAAA --out scan
  python esm_mutscan.py --fasta one.fasta --start 20 --end 45 --plot scan.png
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esm_common import collect_inputs, load_esmc, log
import numpy as np, torch

AAS = "ACDEFGHIKLMNPQRSTVWY"


@torch.inference_mode()
def scan(tok, model, device, seq, start, end):
    aa_ids = {a: tok.convert_tokens_to_ids(a) for a in AAS}
    base = tok(seq, return_tensors="pt")["input_ids"]
    mat = np.zeros((20, end - start), dtype=np.float32)
    for j, pos in enumerate(range(start, end)):
        tpos = pos + 1                                          # +1 for <cls>
        ids = base.clone(); ids[0, tpos] = tok.mask_token_id
        logp = model(input_ids=ids.to(device)).logits[0, tpos].log_softmax(-1)
        wt = base[0, tpos].item()
        for i, a in enumerate(AAS):
            mat[i, j] = float(logp[aa_ids[a]] - logp[wt])
    return mat


def main():
    ap = argparse.ArgumentParser(description="Zero-shot deep mutational scan with ESMC.")
    ap.add_argument("--seq")
    ap.add_argument("--fasta")
    ap.add_argument("--model", default="biohub/ESMC-300M")
    ap.add_argument("--start", type=int, default=0, help="window start (0-based, inclusive)")
    ap.add_argument("--end", type=int, default=-1, help="window end (exclusive; -1 = full length)")
    ap.add_argument("--out", default="mutscan", help="output CSV prefix")
    ap.add_argument("--plot", default=None, help="optional heatmap PNG path")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    inputs = collect_inputs(args.seq, args.fasta)
    name, seq = next(iter(inputs.items()))
    if len(inputs) > 1:
        log(f"[esm_mutscan] multiple inputs; scanning the first ({name})")
    start = max(0, args.start)
    end = len(seq) if args.end < 0 else min(args.end, len(seq))

    tok, model, device = load_esmc(args.model, None if args.device == "auto" else args.device)
    log(f"[esm_mutscan] {name}: positions {start}..{end} on {device}")
    mat = scan(tok, model, device, seq, start, end)

    csv_path = args.out + ".csv"
    with open(csv_path, "w") as fh:
        fh.write("mutant_aa," + ",".join(f"{seq[p]}{p+1}" for p in range(start, end)) + "\n")
        for i, a in enumerate(AAS):
            fh.write(a + "," + ",".join(f"{x:.4f}" for x in mat[i]) + "\n")
    print(f"wrote {csv_path}  (20 x {end-start})")

    if args.plot:
        import matplotlib.pyplot as plt
        vmax = float(np.abs(mat).max())
        fig, ax = plt.subplots(figsize=(max(6, 0.4 * (end - start)), 6))
        im = ax.imshow(mat, aspect="auto", cmap="bwr", vmin=-vmax, vmax=vmax)
        ax.set_yticks(range(20)); ax.set_yticklabels(list(AAS))
        ax.set_xticks(range(end - start)); ax.set_xticklabels([f"{seq[p]}{p+1}" for p in range(start, end)],
                                                              rotation=90, fontsize=7)
        ax.set_title(f"{name}: log P(mut) - log P(wt)  (blue = disfavored)")
        fig.colorbar(im, fraction=0.025, pad=0.02); plt.tight_layout()
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
