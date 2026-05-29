# `autoresearch_silk` — autonomous architecture search for silk-mechanics prediction

A self-contained [Karpathy-style **autoresearch** loop](https://www.verdent.ai/guides/what-is-autoresearch-karpathy):
point a coding agent at this folder and it iterates — *propose an architecture change → run a short
training job → measure mean test R² → keep it if it improved, else roll back* — to learn to predict
silk fiber mechanics (**toughness, E, strength, strain**) **directly from sequence** with ESMC.

Everything an agent needs is in this folder; it should be given **only this folder**.

## Why a search?
Predicting fiber mechanics from a single spidroin sequence is genuinely hard: in the default
`lamm-mit/silkome-masp` (~1028 sequences), many sequences map to the same fiber measurement (~233
distinct property tuples), so the signal is weak. Naive mean-pooling of embeddings leaves a lot on the
table, so we let the agent explore pooling / Conv / attention / multi-task / LoRA / stronger-backbone
ideas and **ratchet** on a fixed metric. (Dataset is a `config.json` switch — e.g. `lamm-mit/silkome-full`.)

## Quick start
```bash
conda activate esm                       # the ESM env (see ../SKILL.md); needs `huggingface-cli login`
python setup.py                          # ONE-TIME: download silkome + cache ESMC-300M embeddings
python run_experiment.py --tag baseline  # the harness: train the model.py architecture → mean test R²
```
`run_experiment.py` is the fixed training/eval harness the loop runs on **every** experiment — it
trains whatever is in `model.py`, computes the metric, and logs it. `--tag baseline` just labels this
first run, which trains the starting architecture (`model.py`'s `MeanPoolMLP`) to set the bar to beat.
Each run prints a **per-epoch `val R²`** progress line and a final `SCALAR  mean test R²`, appends a
row to **`ledger.jsonl`**, and updates **`leaderboard.md`**. To peek at a long run, background it and
tail: `python run_experiment.py --tag x > run.log 2>&1 &` then `tail -f run.log`.

Then hand the folder to a coding agent with the brief in **`program.md`** and let it loop. Or iterate
by hand: edit `model.py` → `python run_experiment.py` → check `leaderboard.md`.

## Full run-through (manual setup → autonomous Codex loop)
The entire sequence, start to finish. Steps 0–4 you do **once, by hand** (so the env, HF auth and the
cache are known-good before the agent takes over); steps 5–8 launch the autonomous loop.

```bash
# 0. PREREQS — the ESM env + HuggingFace auth (silkome is private). Do this in the shell you'll
#    launch the agent from, so the agent inherits both.
conda activate esm
huggingface-cli login                    # paste a token with read access to lamm-mit/silkome-*

# 1. (optional) Two clones: agent experiments in RUN, you make infra fixes in EDIT. See below.
git clone https://github.com/lamm-mit/EvolutionaryScale-protein-mechanics.git ESM-run
cd ESM-run/autoresearch_silk

# 2. ONE-TIME cache: download silkome + cache ESMC-300M embeddings (writes data/ + cache/, gitignored).
python setup.py --smoke-test             # ~30 s pipeline sanity check, no cache written
python setup.py                          # the real cache (downloads the private dataset)

# 3. Put the agent on its own branch so its commit history is isolated.
git checkout -b autoresearch/$(date +%b%d)-silk      # e.g. autoresearch/may29-silk

# 4. BASELINE: train the starting model.py (MeanPoolMLP) to set the bar to beat.
python run_experiment.py --tag baseline  # prints SCALAR mean test R²; writes ledger.jsonl + leaderboard.md

# 5. PER-EXPERIMENT LIMITS (optional but recommended): export once so EVERY run the agent fires
#    inherits them. AR_TIME_BUDGET = wall-clock seconds per run (checked between epochs; 0 = no cap).
export AR_TIME_BUDGET=300                 # 5-min ceiling on each experiment
export AR_EPOCHS=60                       # (optional) also cap epochs per run; overrides config.json
# export AR_MAX_TRAIN=256                 # (optional) train on first N rows only — quick smoke runs

# 6. LAUNCH the agent in full-auto mode (bypasses approvals + sandbox: it needs to run python, make
#    git commits/tags, and reach the network).
codex --yolo                              # (Claude Code: `claude` then accept auto-run, or use a RUN clone)

# 7. PASTE the kickoff prompt below (the "Kick off the agent" block) into the agent and let it loop.

# 8. MONITOR from another shell (same folder):
tail -f run.log 2>/dev/null               # if you background a run: python run_experiment.py ... > run.log 2>&1 &
cat leaderboard.md                        # current best-so-far
tail -f journal.md                        # the agent's running notes / negative results
```

When you want to stop the loop, tell the agent to stop (it's instructed to run "indefinitely"
otherwise). Then harvest results — see **Results & figures** below (`analyze_results.py`,
`export_experiments.py`).

> The agent's prompt (next section) *also* contains steps 0/2/3/4, so if you'd rather not pre-run them
> by hand you can just launch `codex --yolo` and paste — but doing setup yourself first means an
> interactive `huggingface-cli login` (which `--yolo` can't answer) is already handled.

## Kick off the agent (Claude Code / Codex)

Open a coding-agent session **with this folder as the working directory** and paste this prompt. It is
deliberately short — **`program.md` is the full brief** (the git-ratcheted loop, commit/tag steps,
rules, calibration, and idea menu all live there, so they stay in one place):

```text
You are an autonomous ML research agent and this folder is your entire workspace.

GOAL: maximize the mean test R² for predicting four silk-fiber mechanical properties
(toughness, E, strength, strain) directly from protein sequence.

Read program.md — it is your complete brief: the metric, the one-time setup, the exact
git-ratcheted loop (propose → commit+tag → run_experiment.py → keep or revert), the rules,
and the idea menu. Then skim journal.md (what's been tried) and leaderboard.md (best to beat).

Do the one-time setup, run the baseline to set the bar, then run the program.md loop
indefinitely — one change per run, never stop unless asked. When you stop, report the best
architecture and its mean test R².
```

**Two clones (RUN vs EDIT).** Keep optimization experiments separate from reusable-code changes by
cloning the repo twice — the agent experiments and commits per-run in the RUN clone, while infra fixes
are made in the EDIT clone and pulled in:
```bash
git clone https://github.com/lamm-mit/EvolutionaryScale-protein-mechanics.git ESM-edit   # edit infra, push
git clone https://github.com/lamm-mit/EvolutionaryScale-protein-mechanics.git ESM-run    # agent runs/commits here
```
(`data/` and `cache/` are git-ignored, so run `python setup.py` once per clone; the splits/metric are
deterministic so results are comparable.) See `program.md` → "Git workflow".

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
rely on the per-run git commits, so run the loop git-ratcheted (see *program.md*).

## Files
| file | role |
|------|------|
| `program.md` | **the agent's brief**: goal, loop protocol, rules, ideas, caveats |
| `model.py` | **the editable asset** — architecture mapping residue embeddings → 4 targets |
| `config.json` | tunable knobs (lr, epochs, pooling…) + **`esmc_model`** (the backbone) |
| `setup.py` | one-time: cache data + ESMC embeddings (`--model`, `--device` configurable) |
| `dataio.py` | fixed data/metric plumbing (target scaler, grouped split, R²) — don't edit |
| `run_experiment.py` | fixed harness: train → grouped-val early stop → test R² → ledger/leaderboard |
| `leaderboard.md` / `ledger.jsonl` | best-so-far / full history |
| `analyze_results.py` | plots (progress / architecture / params) + tables from `ledger.jsonl` |
| `export_experiments.py` | materialize each committed experiment into a folder (model.py snapshot + diff) |
| `journal.md` | agent's running notes (hypotheses + negative results) |

## Switching dataset or backbone (e.g. on a GPU box / DGX Spark)
Edit `config.json` (`dataset` and/or `esmc_model`), then re-run `setup.py`:
```bash
# smaller MaSp-only dataset:           "dataset": "lamm-mit/silkome-masp"
python setup.py
# stronger backbone:                   "esmc_model": "biohub/ESMC-6B"
python setup.py --model biohub/ESMC-6B --device cuda
python run_experiment.py --tag "6B features"
```
Caches are keyed by **(dataset, model)**, so silkome-full/masp × 300M/600M/6B all coexist.
(`--dataset`/`--model`/`--test-frac` can also be passed on the `setup.py` command line.)

## Notes
- **Private data:** the silkome datasets are private; `data/` and `cache/` are **git-ignored** and never
  committed. Re-run `setup.py` (with HF auth) to repopulate locally.
- **Metric:** mean of the four per-target test R² (scale-invariant).
- **Task scope:** row-level, **one sequence → 4 targets**. The harness is not set up for the
  `silkome-*-idv-grouped` (list-of-sequences-per-fiber) variants — those need a different batcher and a
  `model.py` that consumes a set of sequences per example.
- **Exact-sequence dedup:** `setup.py` drops duplicate sequences (and any test sequence also present in
  train), so the cached silkome-masp split is **891 train / 137 test** vs. the raw HF **895 / 138** —
  this avoids overcounting identical spidroins, and applies to provided splits too.
- **Train/test split** (`config.json` `split_mode`): `auto` (default — the dataset's own `train`/`test`
  if present, else a harness split), `provided`, or `grouped` (pool all splits + dedup + deterministic
  split **grouped by property tuple**, `test_frac`, so identical-fiber sequences don't leak across it).
  Early-stopping always uses a further **grouped** validation split within train. Don't optimize the
  test set directly — prefer changes that also lift grouped-val R².
