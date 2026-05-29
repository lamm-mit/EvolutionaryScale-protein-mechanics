#!/usr/bin/env python
"""FIXED experiment harness for the silk autoresearch loop — do not change the metric/splits.

Trains whatever is in model.py on cached ESMC embeddings, early-stops on a grouped validation
split, evaluates on the held-out test split, logs to ledger.jsonl, updates leaderboard.md, and
prints the SCALAR the agent optimizes: mean test R² over [toughness, E, strength, strain].

  python run_experiment.py            # uses config.json + model.py
  python run_experiment.py --tag "attention pooling v2"
"""
import argparse, json, os, time, datetime
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
import dataio
import model as model_module
from dataio import SilkData, TargetScaler, grouped_train_val_split, r2_per_target, make_batches, TARGETS


def device_auto():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


@torch.inference_mode()
def predict(net, data, idx, scaler, batch_size, device):
    net.eval(); preds = []
    for X, mask, _ in make_batches(data, idx, batch_size, shuffle=False, device=device):
        preds.append(net(X, mask).float().cpu().numpy())
    return scaler.inverse(np.concatenate(preds))         # raw units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="note for the ledger")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    seed = cfg.get("seed", 0)
    np.random.seed(seed); torch.manual_seed(seed)
    device = device_auto() if args.device == "auto" else args.device

    train, test = SilkData("train"), SilkData("test")
    scaler = TargetScaler(train.y)
    tr_idx, val_idx = grouped_train_val_split(
        train.groups, cfg.get("val_frac", 0.15), seed) if cfg.get("group_split", True) else None
    print(f"[run] backbone={cfg['esmc_model']} dim={train.dim} | "
          f"train={len(tr_idx)} val={len(val_idx)} test={len(test)} | device={device}")

    net = model_module.build_model(train.dim, len(TARGETS), cfg).to(device)
    n_params = sum(p.numel() for p in net.parameters())

    # Optional auxiliary supervision (fusion / multi-task): if model.py declares AUX_COLS
    # (e.g. ["family","genus","category1"]) and implements build_aux_heads/auxiliary_loss, we
    # supervise those taxonomy classifiers jointly. Eval/metric are unchanged (forward->4 targets).
    aux_cols = getattr(net, "AUX_COLS", None)
    aux_arrays, aux_w = {}, cfg.get("aux_weight", 0.3)
    if aux_cols:
        counts = {}
        for c in aux_cols:
            arr, classes = train.label_array(c); aux_arrays[c] = arr; counts[c] = len(classes) + 1
        net.build_aux_heads(counts)
        net.to(device)
        print(f"[run] auxiliary heads {counts} (aux_weight={aux_w})")

    opt = torch.optim.Adam(net.parameters(), lr=cfg.get("lr", 5e-4),
                           weight_decay=cfg.get("weight_decay", 1e-4))
    lossfn = torch.nn.MSELoss()

    best_val, best_state, bad = -1e9, None, 0
    t0 = time.time()
    for epoch in range(cfg.get("epochs", 200)):
        net.train()
        for batch in make_batches(train, tr_idx, cfg.get("batch_size", 32), shuffle=True,
                                  seed=seed + epoch, device=device, return_idx=bool(aux_cols)):
            (X, mask, ysub, sub) = batch if aux_cols else (*batch, None)
            yi = torch.tensor(scaler.transform(ysub.cpu().numpy()), device=device)
            opt.zero_grad()
            loss = lossfn(net(X, mask), yi)
            if aux_cols:
                aux_lab = {c: torch.tensor(aux_arrays[c][sub], device=device) for c in aux_cols}
                loss = loss + aux_w * net.auxiliary_loss(X, mask, aux_lab)
            loss.backward()
            if cfg.get("grad_clip"): torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            opt.step()
        _, val_mean = r2_per_target(train.y[val_idx], predict(net, train, val_idx, scaler,
                                                              cfg.get("batch_size", 32), device))
        improved = val_mean > best_val
        if improved:
            best_val, bad = val_mean, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        # progress line (peekable): every epoch when improving, else every 5th epoch
        if improved or epoch % 5 == 0:
            print(f"[run] epoch {epoch:3d}  val R²={val_mean:+.4f}  best={best_val:+.4f}"
                  f"  ({time.time()-t0:.0f}s)" + ("  *new best" if improved else ""), flush=True)
        if bad >= cfg.get("patience", 20):
            print(f"[run] early stop at epoch {epoch} (best val R²={best_val:.4f})", flush=True)
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    test_per, test_mean = r2_per_target(test.y, predict(net, test, np.arange(len(test)),
                                                        scaler, cfg.get("batch_size", 32), device))

    desc = getattr(model_module, "DESCRIPTION", model_module.build_model.__module__)
    entry = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag, "model": desc, "backbone": cfg["esmc_model"], "params": int(n_params),
        "val_r2_mean": round(float(best_val), 4), "test_r2_mean": round(float(test_mean), 4),
        "test_r2": {k: round(v, 4) for k, v in test_per.items()},
        "cfg": {k: cfg[k] for k in ("lr", "epochs", "batch_size", "weight_decay") if k in cfg},
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(HERE, "ledger.jsonl"), "a") as fh:
        fh.write(json.dumps(entry) + "\n")
    update_leaderboard()

    print("\n================  RESULT  ================")
    print(f"  per-target test R²: " + "  ".join(f"{k}={v:+.3f}" for k, v in test_per.items()))
    print(f"  >>> SCALAR  mean test R² = {test_mean:.4f}   (val {best_val:.4f}, {entry['seconds']}s)")
    print("==========================================")


def update_leaderboard():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "ledger.jsonl")) if l.strip()]
    rows.sort(key=lambda r: r["test_r2_mean"], reverse=True)
    lines = ["# Leaderboard — mean test R² (silk mechanics from sequence)\n",
             f"_{len(rows)} experiments logged. Higher is better; baseline-to-beat is the top row._\n",
             "| rank | mean R² | toughness | E | strength | strain | model | backbone | tag |",
             "|----:|----:|----:|----:|----:|----:|------|------|-----|"]
    for i, r in enumerate(rows[:15], 1):
        t = r["test_r2"]
        lines.append(f"| {i} | **{r['test_r2_mean']:.3f}** | {t['toughness']:.3f} | {t['E']:.3f} | "
                     f"{t['strength']:.3f} | {t['strain']:.3f} | {r['model'][:48]} | "
                     f"{r['backbone'].split('/')[-1]} | {r.get('tag','')[:24]} |")
    open(os.path.join(HERE, "leaderboard.md"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
