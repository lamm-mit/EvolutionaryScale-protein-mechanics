"""Export committed experiment snapshots from ledger.jsonl into local folders.

The autoresearch loop stores each experiment as a local git commit and run_experiment.py records
the commit hash in ledger.jsonl. This materializes those committed source snapshots (default
`model.py`) under experiment_snapshots/experiment_000, _001, … plus a best_experiment/ copy.

Adapted from explore-and-discover's export_experiments.py to this problem (metric = mean test R²,
higher is better; editable asset = model.py).

Usage:
    python export_experiments.py
    python export_experiments.py --files model.py config.json
    python export_experiments.py --out-dir experiment_snapshots
"""
from __future__ import annotations
import argparse, csv, json, re, shutil, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = "ledger.jsonl"
OUT_DIR = "experiment_snapshots"
DEFAULT_FILES = ("model.py",)
TARGETS = ["toughness", "E", "strength", "strain"]


def _git(args, cwd=HERE):
    try:
        return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                              text=True, timeout=30).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _repo_paths(files):
    """Map problem-folder-relative files to repo-relative paths (for `git show commit:path`)."""
    prefix = (_git(["rev-parse", "--show-prefix"]) or "").strip()   # e.g. "autoresearch_silk/"
    return [(f"{prefix}{f}", f) for f in files]


def _read_ledger(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        raise SystemExit(f"No experiments in {path}. Run run_experiment.py first.")
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _flat(row, index):
    tr = row.get("test_r2", {})
    flat = {"experiment": index, "commit": row.get("commit", ""), "tag": row.get("tag", ""),
            "model": row.get("model", ""), "backbone": row.get("backbone", ""),
            "params": row.get("params", ""), "val_r2_mean": row.get("val_r2_mean", ""),
            "test_r2_mean": row.get("test_r2_mean", ""), "seconds": row.get("seconds", "")}
    flat.update({f"r2_{t}": tr.get(t, "") for t in TARGETS})
    return flat


def _best_index(rows):
    best, bi = -1e18, None
    for i, r in enumerate(rows):
        v = r.get("test_r2_mean")
        if isinstance(v, (int, float)) and v > best:
            best, bi = v, i
    return bi


def _readme(index, row, exported, missing, is_best):
    tr = row.get("test_r2", {})
    L = [f"# Experiment {index:03d}" + ("  ⭐ BEST" if is_best else ""), "",
         f"- commit: `{row.get('commit','') or 'n/a'}`",
         f"- model: `{row.get('model','?')}`",
         f"- tag: `{row.get('tag','')}`",
         f"- backbone: `{row.get('backbone','')}`  | params: `{row.get('params','')}`",
         f"- **mean test R²: `{row.get('test_r2_mean','?')}`**  (val `{row.get('val_r2_mean','?')}`)",
         f"- per-target R²: " + ", ".join(f"{t}=`{tr.get(t,'?')}`" for t in TARGETS), ""]
    if exported:
        L += ["## Snapshot files", ""] + [f"- `{f}`" for f in exported] + [""]
    if missing:
        L += ["## Missing (no valid commit)", ""] + [f"- `{f}`" for f in missing] + [""]
    return "\n".join(L)


def export(ledger_path, out_dir, files, strict=False):
    if _git(["rev-parse", "--show-toplevel"]) is None:
        raise SystemExit("Not a git repository.")
    rows = _read_ledger(ledger_path)
    repo_files = _repo_paths(files)
    bi = _best_index(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, errors = [], 0

    for index, row in enumerate(rows):
        commit = str(row.get("commit", "")).strip()
        exp = out_dir / f"experiment_{index:03d}"; exp.mkdir(parents=True, exist_ok=True)
        exported, missing = [], []
        if re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
            for repo_path, local in repo_files:
                content = _git(["show", f"{commit}:{repo_path}"])
                if content is None:
                    missing.append(local); errors += 1; continue
                (exp / local).write_text(content)
                exported.append(local)
            patch = _git(["show", "--patch", commit, "--", *[r for r, _ in repo_files]])
            (exp / "changes.patch").write_text(patch or "No patch for these files in this commit.\n")
        else:
            missing = [local for _, local in repo_files]; errors += 1
            (exp / "_export_error.txt").write_text(f"No valid commit for experiment {index}: {commit!r}\n")

        meta = {"experiment": index, "is_best": index == bi, "folder": exp.name,
                "exported_files": exported, "missing_files": missing, "row": row}
        (exp / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        (exp / "README.md").write_text(_readme(index, row, exported, missing, index == bi))
        manifest.append({**_flat(row, index), "is_best": int(index == bi),
                         "folder": exp.name, "missing": ",".join(missing)})

    fields = ["experiment", "is_best", "folder", "commit", "tag", "model", "backbone", "params",
              "val_r2_mean", "test_r2_mean", *[f"r2_{t}" for t in TARGETS], "seconds", "missing"]
    with (out_dir / "manifest.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if bi is not None:
        best_target = out_dir / "best_experiment"
        if best_target.exists():
            shutil.rmtree(best_target)
        shutil.copytree(out_dir / f"experiment_{bi:03d}", best_target)
        summary = _readme(bi, rows[bi], [], [], True)
        (out_dir / "BEST_EXPERIMENT.md").write_text(summary)
        (best_target / "BEST_EXPERIMENT.md").write_text(summary)

    if strict and errors:
        raise SystemExit(f"Export finished with {errors} missing file(s).")
    return len(rows), bi, errors


def main():
    ap = argparse.ArgumentParser(description="Export committed experiment snapshots from ledger.jsonl")
    ap.add_argument("--ledger", default=str(HERE / LEDGER))
    ap.add_argument("--out-dir", default=str(HERE / OUT_DIR))
    ap.add_argument("--files", nargs="+", default=list(DEFAULT_FILES),
                    help="problem-folder-relative files to snapshot per commit (default: model.py)")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    n, bi, errors = export(Path(args.ledger), Path(args.out_dir), args.files, args.strict)
    print(f"Exported {n} experiment folder(s) to {args.out_dir}")
    if bi is not None:
        print(f"Best = experiment_{bi:03d} (mean test R²={_read_ledger(Path(args.ledger))[bi].get('test_r2_mean')}); "
              f"see {Path(args.out_dir)/'BEST_EXPERIMENT.md'}")
    if errors:
        print(f"{errors} file(s) missing (pre-git-tracking runs or bad commit) — see _export_error.txt / metadata.json")


if __name__ == "__main__":
    main()
