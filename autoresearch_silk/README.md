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
Then hand the folder to a coding agent with the brief in **`research.md`** and let it loop. Or iterate
by hand: edit `model.py` → `python run_experiment.py` → check `leaderboard.md`.

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
