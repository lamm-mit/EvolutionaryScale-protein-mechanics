#!/usr/bin/env python
"""esm_train_head — train a predictor on top of ESMC for ANY sequence property.

Two methods:
  * --method head  (default) : ESMC is FROZEN; mean-pooled embeddings feed a small head
                               (LogisticRegression/Ridge, or a torch MLP). Fast, runs anywhere.
  * --method lora            : LoRA adapters are added to ESMC and fine-tuned end-to-end with a
                               classification/regression head (PEFT). More powerful; needs more
                               compute. Adapter + head are saved for `esm_predict.py`.

Task is auto-detected from the target column's dtype (categorical -> classification, numeric ->
regression); override with --task. Data is one dataset (cross-validated) or an explicit --train /
--test split. Each source is a local CSV or a Hugging Face dataset repo (optionally `repo:split`);
if you pass only a HF --train that has a test/validation split, it is used automatically.

Examples
--------
  python esm_train_head.py --out-dir head_model                              # default HF dataset, CV
  python esm_train_head.py --train train.csv --test test.csv --target family
  python esm_train_head.py --train data.csv --seq-column seq --target tm      # numeric -> regression
  python esm_train_head.py --train myorg/my-dataset --target solubility       # auto train/test splits
  python esm_train_head.py --method lora --lora-r 16 --epochs 4 --target family
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from esm_common import load_esmc, embed_sequences, pick_device, log
import numpy as np


# --------------------------------------------------------------------------- data loading
def _is_local(src):
    return os.path.exists(src) or src.lower().endswith(".csv")


def read_one(src, default_split):
    """Load a DataFrame from a local CSV or a HF dataset repo id (optionally `repo:split`)."""
    import pandas as pd
    if _is_local(src):
        return pd.read_csv(src)
    repo_id, _, split = src.partition(":")
    from datasets import load_dataset
    return load_dataset(repo_id, split=split or default_split).to_pandas()


def load_splits(train_src, test_src):
    df_tr = read_one(train_src, "train")
    if test_src:
        return df_tr, read_one(test_src, "test")
    # HF repo without an explicit split: auto-use a test/validation split if one exists
    if not _is_local(train_src) and ":" not in train_src:
        try:
            from datasets import get_dataset_split_names
            avail = get_dataset_split_names(train_src)
            for s in ("test", "validation", "valid", "eval"):
                if s in avail:
                    log(f"[train] using HF split '{s}' as held-out test")
                    return df_tr, read_one(f"{train_src}:{s}", s)
        except Exception:
            pass
    return df_tr, None


def encode_targets(df, target, task, classes=None):
    if task == "classification":
        return np.array([classes.index(str(v)) for v in df[target]])
    return df[target].astype("float32").to_numpy()


# --------------------------------------------------------------------------- LoRA fine-tuning
def run_lora(args, task, df_tr, df_te, classes, meta):
    import torch, torch.nn as nn
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from sklearn.model_selection import train_test_split

    num_labels = len(classes) if task == "classification" else 1
    problem = "regression" if task == "regression" else "single_label_classification"
    device = pick_device(args.device, heavy=("6B" in args.model))
    epochs = args.epochs if args.epochs is not None else 4
    lr = args.lr if args.lr is not None else 1e-4
    targets = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]

    # hold out a validation slice if no explicit test set
    if df_te is None:
        strat = df_tr[args.target] if task == "classification" else None
        df_tr, df_te = train_test_split(df_tr, test_size=0.2, random_state=0, stratify=strat)
        df_tr, df_te = df_tr.reset_index(drop=True), df_te.reset_index(drop=True)
        log("[train] no --test; held out 20% of train for validation")

    tok = AutoTokenizer.from_pretrained(args.model)
    base = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=num_labels, problem_type=problem)
    # NB: no task_type -> generic PeftModel that forwards only our kwargs (ESMC's seq-cls head
    # does not accept the `inputs_embeds` that PeftModelForSequenceClassification would inject).
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules=targets, modules_to_save=["classifier"])
    model = get_peft_model(base, lcfg)
    model = (model.float() if device != "cuda" else model).to(device)
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    def batches(df, shuffle):
        idx = np.arange(len(df))
        if shuffle:
            np.random.default_rng(0).shuffle(idx)
        for i in range(0, len(idx), args.batch_size):
            sub = idx[i:i + args.batch_size]
            seqs = df[args.seq_column].astype(str).iloc[sub].tolist()
            enc = tok(seqs, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_length)
            y = encode_targets(df.iloc[sub], args.target, task, classes)
            yt = (torch.tensor(y, dtype=torch.long) if task == "classification"
                  else torch.tensor(y, dtype=torch.float32).unsqueeze(1))
            yield {k: v.to(device) for k, v in enc.items()}, yt.to(device)

    for ep in range(epochs):
        model.train(); tot = 0.0
        for enc, y in batches(df_tr, True):
            opt.zero_grad()
            out = model(**enc, labels=y)
            out.loss.backward(); opt.step(); tot += out.loss.item()
        log(f"  epoch {ep + 1}/{epochs}  train_loss={tot:.3f}")

    model.eval(); logits = []
    with torch.inference_mode():
        for enc, _ in batches(df_te, False):
            logits.append(model(**enc).logits.float().cpu())
    logits = np.concatenate([t.numpy() for t in logits])
    ytrue = encode_targets(df_te, args.target, task, classes)
    if task == "classification":
        print(f"VALIDATION accuracy: {(logits.argmax(1) == ytrue).mean():.3f}")
    else:
        from sklearn.metrics import r2_score, mean_absolute_error
        pred = logits.squeeze(1)
        print(f"VALIDATION R2: {r2_score(ytrue, pred):.3f}  MAE: {mean_absolute_error(ytrue, pred):.3f}")

    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    meta.update({"num_labels": num_labels, "problem_type": problem, "max_length": args.max_length})


# --------------------------------------------------------------------------- frozen head
def run_head(args, task, df_tr, df_te, classes, meta):
    tok, model, device = load_esmc(args.model, None if args.device == "auto" else args.device)
    log(f"[train] embedding {len(df_tr)} train seq(s) with {args.model} on {device} ...")
    Xtr = np.stack(embed_sequences(tok, model, df_tr[args.seq_column].astype(str).tolist(),
                                   device, "mean", args.batch_size)).astype("float32")
    meta["embed_dim"] = int(Xtr.shape[1])
    ytr = encode_targets(df_tr, args.target, task, classes)
    Xte = yte = None
    if df_te is not None:
        Xte = np.stack(embed_sequences(tok, model, df_te[args.seq_column].astype(str).tolist(),
                                       device, "mean", args.batch_size)).astype("float32")
        yte = encode_targets(df_te, args.target, task, classes)

    from sklearn.preprocessing import StandardScaler
    if args.head == "linear":
        from sklearn.pipeline import make_pipeline
        if task == "classification":
            from sklearn.linear_model import LogisticRegression
            est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)); scoring = None
        else:
            from sklearn.linear_model import Ridge
            est = make_pipeline(StandardScaler(), Ridge(alpha=1.0)); scoring = "r2"
        if Xte is not None:
            est.fit(Xtr, ytr)
            if task == "classification":
                print(f"TEST accuracy: {(est.predict(Xte) == yte).mean():.3f}")
            else:
                from sklearn.metrics import r2_score, mean_absolute_error
                p = est.predict(Xte)
                print(f"TEST R2: {r2_score(yte, p):.3f}  MAE: {mean_absolute_error(yte, p):.3f}")
        elif args.cv:
            from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
            cvs = (StratifiedKFold(args.cv, shuffle=True, random_state=0) if task == "classification"
                   else KFold(args.cv, shuffle=True, random_state=0))
            s = cross_val_score(est, Xtr, ytr, cv=cvs, scoring=scoring)
            print(f"CV {'accuracy' if task=='classification' else 'R2'} ({args.cv}-fold): "
                  f"{s.mean():.3f} +/- {s.std():.3f}")
        est.fit(Xtr, ytr)
        import joblib
        joblib.dump(est, os.path.join(args.out_dir, "head.joblib"))
    else:
        import torch, torch.nn as nn
        from sklearn.model_selection import train_test_split
        epochs = args.epochs if args.epochs is not None else 300
        if Xte is None:
            strat = ytr if task == "classification" else None
            Xtr, Xte, ytr, yte = train_test_split(Xtr, ytr, test_size=0.2, random_state=0, stratify=strat)
        sc = StandardScaler().fit(Xtr)
        meta["scaler_mean"] = sc.mean_.tolist(); meta["scaler_scale"] = sc.scale_.tolist()
        out_dim = len(classes) if task == "classification" else 1
        meta["out_dim"] = out_dim
        Xtr_t = torch.tensor(sc.transform(Xtr), dtype=torch.float32)
        ytr_t = (torch.tensor(ytr, dtype=torch.long) if task == "classification"
                 else torch.tensor(ytr, dtype=torch.float32).unsqueeze(1))
        torch.manual_seed(0)
        head = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(), nn.Dropout(0.3),
                             nn.Linear(128, out_dim))
        opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
        lossfn = nn.CrossEntropyLoss() if task == "classification" else nn.MSELoss()
        for _ in range(epochs):
            head.train(); opt.zero_grad()
            loss = lossfn(head(Xtr_t), ytr_t); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            out = head(torch.tensor(sc.transform(Xte), dtype=torch.float32))
        if task == "classification":
            print(f"TEST accuracy: {(out.argmax(1).numpy() == yte).mean():.3f}")
        else:
            from sklearn.metrics import r2_score, mean_absolute_error
            p = out.squeeze(1).numpy()
            print(f"TEST R2: {r2_score(yte, p):.3f}  MAE: {mean_absolute_error(yte, p):.3f}")
        torch.save(head.state_dict(), os.path.join(args.out_dir, "head.pt"))


def main():
    ap = argparse.ArgumentParser(
        description="Train a head or LoRA adapter on ESMC for any sequence property.")
    ap.add_argument("--train", help="training data: local CSV or HF repo (optionally repo:split)")
    ap.add_argument("--test", help="optional held-out data; else cross-validate / hold out a split")
    ap.add_argument("--repo", default="lamm-mit/structural-protein-families",
                    help="fallback HF dataset when --train is omitted")
    ap.add_argument("--seq-column", default="sequence")
    ap.add_argument("--target", "--label", dest="target", default="family",
                    help="output column (categorical -> classification, numeric -> regression)")
    ap.add_argument("--task", choices=["auto", "classification", "regression"], default="auto")
    ap.add_argument("--method", choices=["head", "lora"], default="head")
    ap.add_argument("--head", choices=["linear", "mlp"], default="linear",
                    help="(head method) linear=LogReg/Ridge, mlp=torch MLP")
    ap.add_argument("--model", default="biohub/ESMC-300M")
    ap.add_argument("--cv", type=int, default=5, help="(head/linear) CV folds when no --test")
    ap.add_argument("--epochs", type=int, default=None, help="MLP head: 300 / LoRA: 4 (if unset)")
    ap.add_argument("--lr", type=float, default=None, help="learning rate (LoRA default 1e-4)")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target-modules", default="layernorm_qkv.1,out_proj,ffn.1,ffn.3",
                    help="comma list (ESMC card defaults)")
    ap.add_argument("--max-length", type=int, default=512, help="(LoRA) token truncation length")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--out-dir", default="head_model")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    import pandas as pd
    df_tr, df_te = load_splits(args.train or args.repo, args.test)
    df_tr = df_tr.dropna(subset=[args.seq_column, args.target])
    if args.max_samples:
        df_tr = df_tr.sample(n=min(args.max_samples, len(df_tr)), random_state=0).reset_index(drop=True)
    if df_te is not None:
        df_te = df_te.dropna(subset=[args.seq_column, args.target])

    task = (args.task if args.task != "auto"
            else ("regression" if pd.api.types.is_numeric_dtype(df_tr[args.target])
                  else "classification"))
    classes = sorted(map(str, df_tr[args.target].unique())) if task == "classification" else None
    os.makedirs(args.out_dir, exist_ok=True)
    meta = {"task": task, "method": args.method, "esmc_model": args.model, "head": args.head,
            "target": args.target, "pool": "mean"}
    if classes is not None:
        meta["classes"] = classes
    log(f"[train] method={args.method} task={task} target='{args.target}' "
        f"n_train={len(df_tr)} n_test={len(df_te) if df_te is not None else 0}")

    if args.method == "lora":
        run_lora(args, task, df_tr, df_te, classes, meta)
    else:
        run_head(args, task, df_tr, df_te, classes, meta)

    json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w"), indent=2)
    print(f"saved {args.method} ({task}) to {args.out_dir}/")


if __name__ == "__main__":
    main()
