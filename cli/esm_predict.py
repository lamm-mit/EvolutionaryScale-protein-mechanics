#!/usr/bin/env python
"""esm_predict — apply a head trained by `esm_train_head.py` to new sequence(s).

Handles both task types saved in meta.json:
  * classification -> predicted label + top-k class probabilities
  * regression     -> predicted numeric value

Examples
--------
  python esm_predict.py --model-dir head_model --seq GGAGQGGYGGLGSQGAGRGGLGGQGAGAAAAAAAA
  python esm_predict.py --model-dir head_reg --fasta unknowns.fasta --out predictions.csv
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esm_common import collect_inputs, load_esmc, embed_sequences
import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Predict with a trained ESMC head.")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--seq")
    ap.add_argument("--fasta")
    ap.add_argument("--out", default=None, help="optional CSV of predictions")
    ap.add_argument("--topk", type=int, default=3, help="classes to show (classification)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.model_dir, "meta.json")))
    task = meta.get("task", "classification")
    method = meta.get("method", "head")
    inputs = collect_inputs(args.seq, args.fasta)
    names, seqs = list(inputs), list(inputs.values())
    proba = yhat = None

    if method == "lora":
        # reload base sequence-classification model + the trained LoRA adapter
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftModel
        from esm_common import pick_device
        device = pick_device(args.device, heavy=("6B" in meta["esmc_model"]))
        tok = AutoTokenizer.from_pretrained(args.model_dir)
        base = AutoModelForSequenceClassification.from_pretrained(
            meta["esmc_model"], num_labels=meta["num_labels"], problem_type=meta["problem_type"])
        model = PeftModel.from_pretrained(base, args.model_dir)
        model = (model.float() if device != "cuda" else model).to(device).eval()
        logits = []
        with torch.inference_mode():
            for i in range(0, len(seqs), 8):
                enc = tok(seqs[i:i + 8], return_tensors="pt", padding=True, truncation=True,
                          max_length=meta.get("max_length", 512)).to(device)
                logits.append(model(**enc).logits.float().cpu())
        logits = torch.cat(logits)
        if task == "classification":
            proba = torch.softmax(logits, 1).numpy()
        else:
            yhat = logits.squeeze(1).numpy()
    else:
        tok, model, device = load_esmc(meta["esmc_model"], None if args.device == "auto" else args.device)
        X = np.stack(embed_sequences(tok, model, seqs, device, pool="mean")).astype("float32")
        if meta["head"] == "linear":
            import joblib
            est = joblib.load(os.path.join(args.model_dir, "head.joblib"))
            proba = est.predict_proba(X) if task == "classification" else None
            yhat = None if task == "classification" else est.predict(X)
        else:
            import torch, torch.nn as nn
            Xs = (X - np.array(meta["scaler_mean"])) / np.array(meta["scaler_scale"])
            out_dim = meta.get("out_dim", len(meta.get("classes", [1])))
            head = nn.Sequential(nn.Linear(meta["embed_dim"], 128), nn.ReLU(), nn.Dropout(0.3),
                                 nn.Linear(128, out_dim))
            head.load_state_dict(torch.load(os.path.join(args.model_dir, "head.pt")))
            head.eval()
            with torch.no_grad():
                out = head(torch.tensor(Xs, dtype=torch.float32))
            proba = torch.softmax(out, 1).numpy() if task == "classification" else None
            yhat = None if task == "classification" else out.squeeze(1).numpy()

    # ---- report ----
    rows = []
    if task == "classification":
        classes = meta["classes"]
        for n, p in zip(names, proba):
            order = np.argsort(p)[::-1][:args.topk]
            top = [(classes[i], float(p[i])) for i in order]
            print(f"{n:24} -> {top[0][0]:14} | " + "  ".join(f"{c}:{v:.2f}" for c, v in top))
            rows.append({"name": n, "prediction": top[0][0],
                         **{f"top{k+1}": f"{c}:{v:.3f}" for k, (c, v) in enumerate(top)}})
    else:
        for n, v in zip(names, yhat):
            print(f"{n:24} -> {meta['target']} = {float(v):.4f}")
            rows.append({"name": n, "prediction": float(v)})

    if args.out:
        import csv
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
