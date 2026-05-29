#!/usr/bin/env python
"""Self-contained LoRA fine-tuning of ESMC for the 4 silk targets (regression).

Unlike the cached-embedding harness, this fine-tunes the backbone end-to-end with LoRA adapters
(PEFT) — more powerful, slower, needs `peft`. Reads data/{train,test}.parquet (made by setup.py),
standardizes targets, trains with a grouped-val early stop, prints mean test R².

  python baselines/lora_finetune.py --model biohub/ESMC-300M --epochs 3 --lora-r 16 --device cuda
"""
import argparse, json, os, sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from dataio import TargetScaler, grouped_train_val_split, r2_per_target, TARGETS
import hashlib, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def groups_of(y):
    keys = [hashlib.md5(np.round(r, 5).tobytes()).hexdigest() for r in y]
    u = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    return np.array([u[k] for k in keys])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=json.load(open(os.path.join(HERE, "config.json")))["esmc_model"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available()
           else "cpu") if args.device == "auto" else args.device
    tr = pd.read_parquet(os.path.join(HERE, "data", "train.parquet"))
    te = pd.read_parquet(os.path.join(HERE, "data", "test.parquet"))
    ytr, yte = tr[TARGETS].to_numpy("float32"), te[TARGETS].to_numpy("float32")
    scaler = TargetScaler(ytr)
    tri, vli = grouped_train_val_split(groups_of(ytr), 0.15, 0)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(TARGETS), problem_type="regression")
    cfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                     target_modules=["layernorm_qkv.1", "out_proj", "ffn.1", "ffn.3"],
                     modules_to_save=["classifier"])
    net = get_peft_model(base, cfg)
    net = (net.float() if dev != "cuda" else net).to(dev)
    net.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=args.lr)

    def batches(df, idx, y, shuffle):
        idx = np.array(idx)
        if shuffle: np.random.default_rng(0).shuffle(idx)
        for i in range(0, len(idx), args.batch_size):
            sub = idx[i:i + args.batch_size]
            enc = tok(df["sequence"].iloc[sub].astype(str).tolist(), return_tensors="pt",
                      padding=True, truncation=True, max_length=args.max_length).to(dev)
            yb = torch.tensor(scaler.transform(y[sub]), dtype=torch.float32, device=dev)
            yield enc, yb

    @torch.inference_mode()
    def predict(df, idx, y):
        net.eval(); out = []
        for enc, _ in batches(df, idx, y, False):
            if dev == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = net(**enc).logits
            else:
                logits = net(**enc).logits
            out.append(logits.float().cpu().numpy())
        return scaler.inverse(np.concatenate(out))

    use_amp = (dev == "cuda")           # bf16 autocast on GPU: ~2x faster, half the memory
    amp = (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)) if use_amp else None
    best, best_state = -1e9, None
    for ep in range(args.epochs):
        net.train(); tot = 0.0
        for enc, yb in batches(tr, tri, ytr, True):
            opt.zero_grad()
            if use_amp:
                with amp():
                    out = net(**enc, labels=yb)
            else:
                out = net(**enc, labels=yb)
            out.loss.backward(); opt.step()
            tot += out.loss.item()
        _, vmean = r2_per_target(ytr[vli], predict(tr, vli, ytr))
        print(f"epoch {ep+1}/{args.epochs} loss={tot:.2f} val_R2={vmean:.4f}", flush=True)
        if vmean > best:
            best = vmean; best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    if best_state: net.load_state_dict(best_state)
    per, mean = r2_per_target(yte, predict(te, np.arange(len(te)), yte))
    print("per-target test R²:", {k: round(v, 3) for k, v in per.items()})
    print(f">>> mean test R² = {mean:.4f}")


if __name__ == "__main__":
    main()
