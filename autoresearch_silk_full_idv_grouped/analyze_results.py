"""Publication-ready plots + cleaned tables from the autoresearch ledger.

Mirrors `explore-and-discover`'s analyze_results.py, adapted to this problem's metric
(**mean test R², higher is better**) and the 4 silk targets. Reads `ledger.jsonl`
(written by run_experiment.py) and writes to `analysis_results/`:

    progress.{png,svg,pdf}                 per-experiment R² + running best (the ratchet curve)
    per_target.{png,svg,pdf}               best run's 4 per-target R² + each target's running best
    architecture_summary.{png,svg,pdf,csv} best R² per model/architecture
    parameter_vs_performance.{png,svg,pdf,csv}  params vs R² (by backbone)
    results_clean.csv                      flattened ledger
    summary.json                           best experiment + run counts

Usage:  python analyze_results.py [--ledger ledger.jsonl] [--out-dir analysis_results]
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
_CACHE = HERE / ".cache"; (_CACHE / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGETS = ["toughness", "E", "strength", "strain"]


def save_all(fig, out_dir, prefix):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"{prefix}.{ext}", bbox_inches="tight")
    plt.close(fig)


def load_ledger(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        tr = d.get("test_r2", {})
        rows.append({
            "experiment": len(rows),
            "time": d.get("time", ""), "tag": d.get("tag", ""), "commit": d.get("commit", ""),
            "status": d.get("status", ""),
            "model": d.get("model", "?"), "backbone": str(d.get("backbone", "")).split("/")[-1],
            "params": d.get("params", np.nan),
            "val_r2_mean": d.get("val_r2_mean", np.nan), "test_r2_mean": d.get("test_r2_mean", np.nan),
            **{f"r2_{t}": tr.get(t, np.nan) for t in TARGETS},
            "seconds": d.get("seconds", np.nan),
        })
    return pd.DataFrame(rows)


def plot_progress(df, out_dir):
    best = np.maximum.accumulate(df["test_r2_mean"].values)
    bi = int(np.argmax(df["test_r2_mean"].values))
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.axhline(0, color="0.6", lw=1, ls=":")                      # "predict the mean" floor
    ax.plot(df["experiment"], df["val_r2_mean"], "o-", color="tab:gray", alpha=0.5,
            ms=4, label="val R² (per run)")
    ax.plot(df["experiment"], df["test_r2_mean"], "o-", color="tab:blue", ms=5, label="test R² (per run)")
    ax.step(df["experiment"], best, where="post", color="tab:red", lw=2, label="best test R² so far")
    ax.scatter([df["experiment"][bi]], [df["test_r2_mean"][bi]], s=180, marker="*",
               color="gold", edgecolors="k", zorder=5, label=f"best = {df['test_r2_mean'][bi]:.3f}")
    ax.set_xlabel("experiment #"); ax.set_ylabel("mean test R²  (higher = better)")
    ax.set_title("Autoresearch progress - grouped silk mechanics from sequence sets")
    ax.legend(loc="best", fontsize=9)
    save_all(fig, out_dir, "progress")
    return bi


def plot_per_target(df, out_dir, bi):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    best = df.iloc[bi]
    a1.bar(TARGETS, [best[f"r2_{t}"] for t in TARGETS], color="teal")
    a1.axhline(0, color="0.5", lw=1); a1.set_ylabel("test R²")
    a1.set_title(f"Best run per-target R²\n({best['model'][:40]}, mean={best['test_r2_mean']:.3f})")
    for t in TARGETS:
        a2.plot(df["experiment"], np.maximum.accumulate(df[f"r2_{t}"].fillna(-9).values), label=t)
    a2.axhline(0, color="0.5", lw=1, ls=":"); a2.set_xlabel("experiment #")
    a2.set_ylabel("running-best per-target R²"); a2.set_title("Per-target best over time"); a2.legend(fontsize=8)
    save_all(fig, out_dir, "per_target")


def plot_architectures(df, out_dir):
    grp = df.groupby("model")["test_r2_mean"].max().sort_values(ascending=False)
    grp.to_csv(out_dir / "architecture_summary.csv", header=["best_test_r2_mean"])
    fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * len(grp))))
    ax.barh([m[:50] for m in grp.index][::-1], grp.values[::-1], color="tab:purple")
    ax.axvline(0, color="0.5", lw=1); ax.set_xlabel("best mean test R²")
    ax.set_title("Best result per architecture"); save_all(fig, out_dir, "architecture_summary")


def plot_param_perf(df, out_dir):
    d = df.dropna(subset=["params", "test_r2_mean"])
    d.to_csv(out_dir / "parameter_vs_performance.csv", index=False)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    for bb, sub in d.groupby("backbone"):
        ax.scatter(sub["params"], sub["test_r2_mean"], s=60, alpha=0.8, label=bb)
    ax.axhline(0, color="0.5", lw=1, ls=":"); ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log)"); ax.set_ylabel("mean test R²")
    ax.set_title("Parameters vs performance"); ax.legend(title="backbone", fontsize=8)
    save_all(fig, out_dir, "parameter_vs_performance")


def main():
    ap = argparse.ArgumentParser(description="Plot autoresearch results from ledger.jsonl")
    ap.add_argument("--ledger", default=str(HERE / "ledger.jsonl"))
    ap.add_argument("--out-dir", default=str(HERE / "analysis_results"))
    args = ap.parse_args()
    if not os.path.exists(args.ledger) or os.path.getsize(args.ledger) == 0:
        raise SystemExit(f"No experiments in {args.ledger}. Run run_experiment.py first.")
    df_all = load_ledger(args.ledger)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out / "results_clean.csv", index=False)   # full trace, incl crashes

    # best/progress/plots use only SCORED rows (drop crashes + any unscored/NaN test_r2_mean)
    scored = (df_all["status"].astype(str) != "crash") & df_all["test_r2_mean"].apply(
        lambda v: isinstance(v, (int, float)) and np.isfinite(v))
    df = df_all[scored].reset_index(drop=True)
    n_crash = int((df_all["status"].astype(str) == "crash").sum())
    if df.empty:
        raise SystemExit(f"No scored experiments yet ({len(df_all)} rows, {n_crash} crashes). "
                         "Run a successful run_experiment.py first.")

    bi = plot_progress(df, out)
    plot_per_target(df, out, bi)
    plot_architectures(df, out)
    plot_param_perf(df, out)

    best = df.iloc[bi]
    json.dump({
        "n_experiments": int(len(df_all)), "n_scored": int(len(df)), "n_crashes": n_crash,
        "best_test_r2_mean": float(best["test_r2_mean"]),
        "best_model": best["model"], "best_tag": best["tag"], "best_commit": best["commit"],
        "best_per_target": {t: float(best[f"r2_{t}"]) for t in TARGETS},
        "first_baseline_r2": float(df.iloc[0]["test_r2_mean"]),
    }, open(out / "summary.json", "w"), indent=2)

    print(f"{len(df)} scored experiments ({len(df_all)} total, {n_crash} crashes) -> {out}/")
    print(f"best: {best['test_r2_mean']:.4f}  ({best['model'][:50]}, tag='{best['tag']}', "
          f"commit {best['commit'] or 'n/a'})")
    print("wrote progress / per_target / architecture_summary / parameter_vs_performance (png/svg/pdf)"
          " + results_clean.csv + summary.json")


if __name__ == "__main__":
    main()
