#!/usr/bin/env python
"""Fixed grouped autoresearch harness.

Trains model.py on cached set-of-sequences ESMC embeddings, early-stops on a grouped validation
split, evaluates on the held-out idv-grouped test split, and logs mean test R².
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import time

import numpy as np
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
import model as model_module
from dataio import (
    GroupedSilkData,
    RAW_TARGETS,
    TargetScaler,
    grouped_train_val_split,
    make_batches,
    r2_per_target,
)

LEDGER = os.path.join(HERE, "ledger.jsonl")


def device_auto():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.inference_mode()
def predict(net, data, idx, scaler, batch_size, device):
    net.eval()
    preds = []
    for X, seq_mask, cat_ids, seq_lengths, _ in make_batches(data, idx, batch_size, shuffle=False, device=device):
        preds.append(net(X, seq_mask, cat_ids, seq_lengths).float().cpu().numpy())
    return scaler.inverse(np.concatenate(preds))


def _append(entry):
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def _prior_best():
    best = float("-inf")
    if os.path.exists(LEDGER):
        for line in open(LEDGER):
            line = line.strip()
            if not line:
                continue
            try:
                v = json.loads(line).get("test_r2_mean")
            except Exception:
                continue
            if isinstance(v, (int, float)) and v > best:
                best = v
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="note for the ledger")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(HERE, "config.json")))
    seed = cfg.get("seed", 0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = device_auto() if args.device == "auto" else args.device

    epochs = int(os.environ.get("AR_EPOCHS", cfg.get("epochs", 200)))
    time_budget = float(os.environ.get("AR_TIME_BUDGET", "0") or 0)
    max_train = int(os.environ.get("AR_MAX_TRAIN", "0") or 0)

    try:
        commit = subprocess.check_output(
            ["git", "-C", HERE, "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = ""

    base = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "commit": commit,
        "dataset": cfg.get("dataset", "lamm-mit/silkome-full-idv-grouped"),
        "target_mode": cfg.get("target_mode", "raw"),
        "backbone": cfg["esmc_model"],
    }

    try:
        train, test = GroupedSilkData("train"), GroupedSilkData("test")
        scaler = TargetScaler(train.y)
        tr_idx, val_idx = grouped_train_val_split(train.groups, cfg.get("val_frac", 0.15), seed)
        if max_train:
            tr_idx = tr_idx[:max_train]

        print(
            f"[run] backbone={cfg['esmc_model']} dim={train.dim} | groups train={len(tr_idx)} "
            f"val={len(val_idx)} test={len(test)} | categories={train.n_categories - 1} "
            f"| device={device} | epochs={epochs}"
            + (f" | budget={time_budget:.0f}s" if time_budget else ""),
            flush=True,
        )

        net = model_module.build_model(
            train.dim, len(RAW_TARGETS), cfg, n_categories=train.n_categories
        ).to(device)
        n_params = sum(p.numel() for p in net.parameters())
        opt = torch.optim.Adam(net.parameters(), lr=cfg.get("lr", 5e-4), weight_decay=cfg.get("weight_decay", 1e-4))
        lossfn = torch.nn.MSELoss()

        best_val, best_state, bad, stop = -1e9, None, 0, "max-epochs"
        t0 = time.time()
        for epoch in range(epochs):
            net.train()
            for X, seq_mask, cat_ids, seq_lengths, ysub in make_batches(
                train, tr_idx, cfg.get("batch_size", 32), shuffle=True, seed=seed + epoch, device=device
            ):
                yi = torch.tensor(scaler.transform(ysub.cpu().numpy()), device=device)
                opt.zero_grad()
                loss = lossfn(net(X, seq_mask, cat_ids, seq_lengths), yi)
                loss.backward()
                if cfg.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
                opt.step()

            _, val_mean = r2_per_target(
                train.y[val_idx],
                predict(net, train, val_idx, scaler, cfg.get("batch_size", 32), device),
            )
            improved = val_mean > best_val
            if improved:
                best_val, bad = val_mean, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
            if improved or epoch % 5 == 0:
                print(
                    f"[run] epoch {epoch:3d}  val R²={val_mean:+.4f}  best={best_val:+.4f}"
                    f"  ({time.time() - t0:.0f}s)" + ("  *new best" if improved else ""),
                    flush=True,
                )
            if bad >= cfg.get("patience", 20):
                stop = "early-stop"
                print(f"[run] early stop at epoch {epoch}", flush=True)
                break
            if time_budget and (time.time() - t0) > time_budget:
                stop = "time-budget"
                print(f"[run] hit AR_TIME_BUDGET={time_budget:.0f}s at epoch {epoch}", flush=True)
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        test_per, test_mean = r2_per_target(
            test.y, predict(net, test, np.arange(len(test)), scaler, cfg.get("batch_size", 32), device)
        )
        status = "keep" if test_mean > _prior_best() else "discard"
        desc = getattr(model_module, "DESCRIPTION", "model.py")
        entry = {
            **base,
            "status": status,
            "model": desc,
            "params": int(n_params),
            "val_r2_mean": round(float(best_val), 4),
            "test_r2_mean": round(float(test_mean), 4),
            "test_r2": {k: round(v, 4) for k, v in test_per.items()},
            "stop": stop,
            "cfg": {**{k: cfg[k] for k in ("lr", "batch_size", "weight_decay") if k in cfg}, "epochs": epochs},
            "seconds": round(time.time() - t0, 1),
        }
        _append(entry)
        update_leaderboard()

        print("\n================  RESULT  ================")
        print("  per-target test R²: " + "  ".join(f"{k}={v:+.3f}" for k, v in test_per.items()))
        print(f"  status: {status.upper()}  (stop: {stop})")
        print(f"  >>> SCALAR  mean test R² = {test_mean:.4f}   (val {best_val:.4f}, {entry['seconds']}s)")
        print("==========================================")

    except Exception as e:
        _append({**base, "status": "crash", "error": repr(e)[:300]})
        raise


def update_leaderboard():
    rows = []
    if os.path.exists(LEDGER):
        rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
    rows = [r for r in rows if isinstance(r.get("test_r2_mean"), (int, float)) and r.get("test_r2")]
    rows.sort(key=lambda r: r["test_r2_mean"], reverse=True)
    lines = [
        "# Leaderboard - mean test R2 (grouped silk mechanics from sequence sets)\n",
        f"_{len(rows)} scored experiments. Higher is better; baseline-to-beat is the top row._\n",
        "| rank | mean R2 | toughness | E | strength | strain | model | backbone | tag |",
        "|----:|----:|----:|----:|----:|----:|------|------|-----|",
    ]
    for i, r in enumerate(rows[:15], 1):
        t = r["test_r2"]
        lines.append(
            f"| {i} | **{r['test_r2_mean']:.3f}** | {t['toughness']:.3f} | {t['E']:.3f} | "
            f"{t['strength']:.3f} | {t['strain']:.3f} | {r['model'][:48]} | "
            f"{r['backbone'].split('/')[-1]} | {r.get('tag','')[:24]} |"
        )
    open(os.path.join(HERE, "leaderboard.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
