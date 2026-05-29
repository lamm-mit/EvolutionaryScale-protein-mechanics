# `autoresearch_silk` — autonomous architecture search for silk-mechanics prediction

A self-contained [Karpathy-style **autoresearch** loop](https://www.verdent.ai/guides/what-is-autoresearch-karpathy):
point a coding agent at this folder and it iterates — *propose an architecture change → run a short
training job → measure mean test R² → keep it if it improved, else roll back* — to learn to predict
silk fiber mechanics (**toughness, E, strength, strain**) **directly from sequence** with ESMC.

Everything an agent needs is in this folder; it should be given **only this folder**.

## Why a search?
Predicting fiber mechanics from a single spidroin sequence is genuinely hard: in `lamm-mit/silkome-full`,
~3170 distinct sequences map to only ~268 distinct property tuples (properties are per-fiber; many
sequences share one measurement). Naive mean-pooling of embeddings leaves a lot on the table, so we let
the agent explore pooling / Conv / attention / multi-task / LoRA / stronger-backbone ideas and **ratchet**
on a fixed, leakage-controlled metric.

## Quick start
```bash
conda activate esm                       # the ESM env (see ../SKILL.md); needs `huggingface-cli login`
python setup.py                          # ONE-TIME: download silkome + cache ESMC-300M embeddings
python run_experiment.py --tag baseline  # train model.py, print mean test R², update leaderboard
```
Each run prints a **per-epoch `val R²`** progress line and a final `SCALAR  mean test R²`, appends a
row to **`ledger.jsonl`**, and updates **`leaderboard.md`**. To peek at a long run, background it and
tail: `python run_experiment.py --tag x > run.log 2>&1 &` then `tail -f run.log`.

Then hand the folder to a coding agent with the brief in **`research.md`** and let it loop. Or iterate
by hand: edit `model.py` → `python run_experiment.py` → check `leaderboard.md`.

## Kick off the agent (Claude Code / Codex)

Open a coding-agent session **with this folder as the working directory** and paste this prompt:

```text
You are an autonomous ML research agent; this folder is your entire workspace.

GOAL: maximize the mean test R² for predicting four silk-fiber mechanical properties
(toughness, E, strength, strain) directly from protein sequence, using cached ESMC embeddings.

FIRST, read research.md (your full brief: rules, metric, calibration, idea menu), then journal.md
(what's been tried — don't repeat dead ends) and leaderboard.md (current best to beat).

ONE-TIME, only if a cache/ folder is missing:  conda activate esm && python setup.py
(needs `huggingface-cli login`; the silkome dataset is private).

THEN loop, indefinitely:
  1. Form ONE hypothesis likely to raise mean test R².
  2. Edit ONLY model.py (and optionally config.json); keep the contract in model.py's header.
     Do NOT modify dataio.py, run_experiment.py, data/, or cache/.
  3. Run:  python run_experiment.py --tag "<short idea>"   and read the printed
     `SCALAR  mean test R²`.
  4. RATCHET WITH GIT (best model.py = git HEAD): if it beat the top of leaderboard.md, note it in
     journal.md and `git add -A && git commit -m "exp: <idea> | mean test R²=<value>"`. If it did NOT
     improve, `git restore model.py config.json` (revert to the last good architecture), append the
     negative result to journal.md, and `git commit -m "rejected: <idea>"` (journal/ledger only).
  5. Repeat. One change per run. No test peeking beyond that scalar; prefer changes that also
     lift the grouped-val R² (memorizing the small test set is not progress).

CONTEXT: this is genuinely hard — every baseline (mean-pool / Ridge / RF / silk-type-mean /
attention / conv / LoRA) sits at R² ≈ 0 ("predict the mean"). Any clearly positive, repeatable
mean test R² is a real result. Explore the optional directions in research.md (sequence-aware
pooling/conv/transformer, taxonomy fusion via the AUX hook, sequence-pattern features, log-targets /
multi-task, LoRA fine-tuning in baselines/, or a stronger backbone via config.json) — or invent your
own. Keep iterating; when you stop, report the best architecture and its mean test R².
```

**Two clones (RUN vs EDIT).** Keep optimization experiments separate from reusable-code changes by
cloning the repo twice — the agent experiments and commits per-run in the RUN clone, while infra fixes
are made in the EDIT clone and pulled in:
```bash
git clone https://github.com/lamm-mit/EvolutionaryScale-protein-mechanics.git ESM-edit   # edit infra, push
git clone https://github.com/lamm-mit/EvolutionaryScale-protein-mechanics.git ESM-run    # agent runs/commits here
```
(`data/` and `cache/` are git-ignored, so run `python setup.py` once per clone; the splits/metric are
deterministic so results are comparable.) See `research.md` → "Git workflow".

To run a *stronger backbone* mid-search, the agent edits `config.json` (`esmc_model`) and re-runs
`python setup.py --model <hf_id> --device cuda`.

## Results & figures

After some experiments have been logged, render publication-ready plots (à la
`explore-and-discover`):
```bash
python analyze_results.py        # reads ledger.jsonl -> analysis_results/
```
Writes `analysis_results/`: **`progress.{png,svg,pdf}`** (per-run R² + the running-best ratchet
curve), **`per_target.*`** (the best run's 4 per-target R² + each target's best over time),
**`architecture_summary.*` (+.csv)** (best R² per architecture), **`parameter_vs_performance.*`
(+.csv)** (params vs R², by backbone), plus `results_clean.csv` and `summary.json`. (Outputs are
git-ignored.)

To **materialize every committed experiment** as a folder (recovering each one's `model.py` snapshot
from its git commit):
```bash
python export_experiments.py              # ledger.jsonl -> experiment_snapshots/
```
Writes `experiment_snapshots/experiment_NNN/` (each with `model.py`, `changes.patch`, `metadata.json`,
`README.md`), a `best_experiment/` copy, `BEST_EXPERIMENT.md`, and `manifest.{tsv,json}`. Snapshots
rely on the per-run git commits, so run the loop git-ratcheted (see *research.md*).

## Files
| file | role |
|------|------|
| `research.md` | **the agent's brief**: goal, loop protocol, rules, ideas, caveats |
| `model.py` | **the editable asset** — architecture mapping residue embeddings → 4 targets |
| `config.json` | tunable knobs (lr, epochs, pooling…) + **`esmc_model`** (the backbone) |
| `setup.py` | one-time: cache data + ESMC embeddings (`--model`, `--device` configurable) |
| `dataio.py` | fixed data/metric plumbing (target scaler, grouped split, R²) — don't edit |
| `run_experiment.py` | fixed harness: train → grouped-val early stop → test R² → ledger/leaderboard |
| `leaderboard.md` / `ledger.jsonl` | best-so-far / full history |
| `analyze_results.py` | plots (progress / architecture / params) + tables from `ledger.jsonl` |
| `export_experiments.py` | materialize each committed experiment into a folder (model.py snapshot + diff) |
| `journal.md` | agent's running notes (hypotheses + negative results) |
| `baselines/` | drop-in architectures (attention, conv) + a self-contained LoRA script |

## Switching backbone (e.g. on a GPU box / DGX Spark)
```bash
# edit config.json: "esmc_model": "biohub/ESMC-6B", then:
python setup.py --model biohub/ESMC-6B --device cuda
python run_experiment.py --tag "6B features"
```
Caches are per-model, so 300M / 600M / 6B can coexist.

## Notes
- **Private data:** `lamm-mit/silkome-full` is private; `data/` and `cache/` are **git-ignored** and never
  committed. Re-run `setup.py` (with HF auth) to repopulate locally.
- **Metric:** mean of the four per-target test R² (scale-invariant). The validation split used for early
  stopping is **grouped by property tuple** to avoid identical-fiber leakage; the test split is the
  dataset's own. Don't optimize the test set directly — prefer changes that also lift grouped-val R².
