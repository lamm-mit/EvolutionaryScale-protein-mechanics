#!/usr/bin/env python
"""FIXED experiment harness for the silk autoresearch loop — do not change the metric/splits.

Trains whatever is in model.py on cached ESMC embeddings, early-stops on a grouped validation
split, evaluates on the held-out test split, logs to ledger.jsonl, updates leaderboard.md, and
prints the SCALAR the agent optimizes: mean test R² over [toughness, E, strength, strain].

  python run_experiment.py            # uses config.json + model.py
  python run_experiment.py --tag "attention pooling v2"

Env overrides (handy for smoke tests / bounding long runs; do NOT use for comparable results):
  AR_EPOCHS=5         cap training epochs
  AR_MAX_TRAIN=256    use only the first N training rows
  AR_TIME_BUDGET=120  stop training after N seconds of wall-clock (post-warmup)
"""
import argparse, json, os, time, datetime, subprocess
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
import dataio
import model as model_module
from dataio import SilkData, TargetScaler, grouped_train_val_split, r2_per_target, make_batches, TARGETS

LEDGER = os.path.join(HERE, "ledger.jsonl")


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
    np.random.seed(seed); torch.manual_seed(seed)
    device = device_auto() if args.device == "auto" else args.device

    epochs = int(os.environ.get("AR_EPOCHS", cfg.get("epochs", 200)))
    time_budget = float(os.environ.get("AR_TIME_BUDGET", "0") or 0)     # seconds; 0 = no cap
    max_train = int(os.environ.get("AR_MAX_TRAIN", "0") or 0)           # 0 = all rows

    # Record the current commit so analyze/export can recover the code. As in the original
    # autoresearch, this assumes the agent has committed the edit BEFORE running (see program.md);
    # it is not validated here.
    try:
        commit = subprocess.check_output(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = ""
    base = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "tag": args.tag,
            "commit": commit, "backbone": cfg["esmc_model"]}

    try:
        train, test = SilkData("train"), SilkData("test")
        scaler = TargetScaler(train.y)
        val_frac = cfg.get("val_frac", 0.15)
        if cfg.get("group_split", True):                 # default: leakage-safe (group by fiber tuple)
            tr_idx, val_idx = grouped_train_val_split(train.groups, val_frac, seed)
            split_kind = "grouped"
        else:                                            # plain random val split (group_split=false)
            perm = np.random.default_rng(seed).permutation(len(train))
            n_val = max(1, int(round(val_frac * len(train))))
            val_idx, tr_idx = perm[:n_val], perm[n_val:]
            split_kind = "random"
        if max_train:
            tr_idx = tr_idx[:max_train]
        print(f"[run] backbone={cfg['esmc_model']} dim={train.dim} | train={len(tr_idx)} "
              f"val={len(val_idx)} ({split_kind}) test={len(test)} | device={device} | epochs={epochs}"
              + (f" | budget={time_budget:.0f}s" if time_budget else ""), flush=True)

        net = model_module.build_model(train.dim, len(TARGETS), cfg).to(device)
        n_params = sum(p.numel() for p in net.parameters())

        # Optional auxiliary supervision (fusion / multi-task): model.py may declare AUX_COLS +
        # build_aux_heads/auxiliary_loss to jointly train taxonomy classifiers. Eval is unchanged.
        aux_cols = getattr(net, "AUX_COLS", None)
        aux_arrays, aux_w = {}, cfg.get("aux_weight", 0.3)
        if aux_cols:
            counts = {}
            for c in aux_cols:
                arr, classes = train.label_array(c); aux_arrays[c] = arr; counts[c] = len(classes) + 1
            net.build_aux_heads(counts); net.to(device)
            print(f"[run] auxiliary heads {counts} (aux_weight={aux_w})", flush=True)

        opt = torch.optim.Adam(net.parameters(), lr=cfg.get("lr", 5e-4),
                               weight_decay=cfg.get("weight_decay", 1e-4))
        lossfn = torch.nn.MSELoss()

        best_val, best_state, bad, stop = -1e9, None, 0, "max-epochs"
        t0 = time.time()
        for epoch in range(epochs):
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
                if cfg.get("grad_clip"):
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
                opt.step()
            _, val_mean = r2_per_target(train.y[val_idx], predict(net, train, val_idx, scaler,
                                                                  cfg.get("batch_size", 32), device))
            improved = val_mean > best_val
            if improved:
                best_val, bad = val_mean, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
            if improved or epoch % 5 == 0:
                print(f"[run] epoch {epoch:3d}  val R²={val_mean:+.4f}  best={best_val:+.4f}"
                      f"  ({time.time()-t0:.0f}s)" + ("  *new best" if improved else ""), flush=True)
            if bad >= cfg.get("patience", 20):
                stop = "early-stop"; print(f"[run] early stop at epoch {epoch}", flush=True); break
            if time_budget and (time.time() - t0) > time_budget:
                stop = "time-budget"
                print(f"[run] hit AR_TIME_BUDGET={time_budget:.0f}s at epoch {epoch}", flush=True); break

        if best_state is not None:
            net.load_state_dict(best_state)
        test_per, test_mean = r2_per_target(test.y, predict(net, test, np.arange(len(test)),
                                                            scaler, cfg.get("batch_size", 32), device))
        status = "keep" if test_mean > _prior_best() else "discard"
        desc = getattr(model_module, "DESCRIPTION", "model.py")
        entry = {**base, "status": status, "model": desc, "params": int(n_params),
                 "val_r2_mean": round(float(best_val), 4), "test_r2_mean": round(float(test_mean), 4),
                 "test_r2": {k: round(v, 4) for k, v in test_per.items()}, "stop": stop,
                 "split": split_kind,
                 "cfg": {**{k: cfg[k] for k in ("lr", "batch_size", "weight_decay") if k in cfg},
                         "epochs": epochs}, "seconds": round(time.time() - t0, 1)}
        _append(entry); update_leaderboard()

        print("\n================  RESULT  ================")
        print(f"  per-target test R²: " + "  ".join(f"{k}={v:+.3f}" for k, v in test_per.items()))
        print(f"  status: {status.upper()}  (stop: {stop})")
        print(f"  >>> SCALAR  mean test R² = {test_mean:.4f}   (val {best_val:.4f}, {entry['seconds']}s)")
        print("==========================================")

    except Exception as e:                                   # log crashes so they aren't retried blindly
        _append({**base, "status": "crash", "error": repr(e)[:300]})
        raise


def update_leaderboard():
    rows = [json.loads(l) for l in open(LEDGER) if l.strip()]
    rows = [r for r in rows if isinstance(r.get("test_r2_mean"), (int, float)) and r.get("test_r2")]
    rows.sort(key=lambda r: r["test_r2_mean"], reverse=True)
    lines = ["# Leaderboard — mean test R² (silk mechanics from sequence)\n",
             f"_{len(rows)} scored experiments. Higher is better; baseline-to-beat is the top row._\n",
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
