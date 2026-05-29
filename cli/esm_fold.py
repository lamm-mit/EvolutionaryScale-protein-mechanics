#!/usr/bin/env python
"""esm_fold — predict 3-D structure(s) with ESMFold2 (local) and save PDB/mmCIF.

Examples
--------
  # one sequence from the command line -> folds/query.pdb + .cif
  python esm_fold.py --seq MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG

  # many sequences from a FASTA -> one structure file per record + summary.csv
  python esm_fold.py --fasta proteins.fasta --out folds --formats pdb,cif

Outputs per record: <out>/<name>.pdb and/or .cif, plus <out>/fold_summary.csv (pLDDT, pTM, length).
Runs on CPU in float32 by default (Apple-Silicon MPS lacks ops ESMFold2 needs); uses CUDA if present.
"""
import argparse, csv, os, sys
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esm_common import collect_inputs, pick_device, log
import torch


def cif_to_pdb(cif_text: str, pdb_path: str):
    """Convert an mmCIF string to a PDB file using biotite."""
    import io
    import biotite.structure.io.pdbx as pdbx
    import biotite.structure.io.pdb as pdb
    cif = pdbx.CIFFile.read(io.StringIO(cif_text))
    structure = pdbx.get_structure(cif, model=1)
    pf = pdb.PDBFile()
    pdb.set_structure(pf, structure)
    pf.write(pdb_path)


def main():
    ap = argparse.ArgumentParser(description="Fold protein(s) with ESMFold2 -> PDB/mmCIF.")
    src = ap.add_argument_group("input (one or both)")
    src.add_argument("--seq", help="a single amino-acid sequence")
    src.add_argument("--fasta", help="FASTA file with one or more sequences")
    ap.add_argument("--out", default="folds", help="output directory (default: folds)")
    ap.add_argument("--formats", default="pdb,cif", help="comma list: pdb,cif (default both)")
    ap.add_argument("--model", default="biohub/ESMFold2-Fast",
                    choices=["biohub/ESMFold2-Fast", "biohub/ESMFold2"])
    ap.add_argument("--loops", type=int, default=3, help="trunk recycles (default 3)")
    ap.add_argument("--steps", type=int, default=50, help="diffusion sampling steps (default 50)")
    ap.add_argument("--diffusion-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    formats = {f.strip().lower() for f in args.formats.split(",") if f.strip()}
    inputs = collect_inputs(args.seq, args.fasta)
    os.makedirs(args.out, exist_ok=True)
    device = pick_device(args.device, heavy=True)
    log(f"[esm_fold] {len(inputs)} sequence(s) on {device} with {args.model}")

    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput

    model = ESMFold2Model.from_pretrained(args.model)
    model = (model.float() if device != "cuda" else model).to(device).eval()
    amp = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if device == "cuda" else nullcontext())

    rows = []
    for name, seq in inputs.items():
        spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)])
        with torch.inference_mode(), amp:
            res = ESMFold2InputBuilder().fold(
                model, spi, num_loops=args.loops, num_sampling_steps=args.steps,
                num_diffusion_samples=args.diffusion_samples, seed=args.seed)
        plddt, ptm = float(res.plddt.mean()), float(res.ptm)
        cif_text = res.complex.to_mmcif()
        base = os.path.join(args.out, name)
        if "cif" in formats:
            open(base + ".cif", "w").write(cif_text)
        if "pdb" in formats:
            try:
                cif_to_pdb(cif_text, base + ".pdb")
            except Exception as e:
                log(f"  [warn] PDB conversion failed for {name}: {e}; wrote .cif only")
                open(base + ".cif", "w").write(cif_text)
        log(f"  {name}: len={len(seq)} pLDDT={plddt:.3f} pTM={ptm:.3f}")
        rows.append({"name": name, "length": len(seq), "pLDDT": round(plddt, 4),
                     "pTM": round(ptm, 4)})

    summ = os.path.join(args.out, "fold_summary.csv")
    with open(summ, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "length", "pLDDT", "pTM"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} structure(s) to {args.out}/ and {summ}")


if __name__ == "__main__":
    main()
