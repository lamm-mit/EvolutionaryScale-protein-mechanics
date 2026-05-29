# Research goal — predict silk fiber mechanics from sequence

You are an autonomous ML research agent. **Maximize the mean test R²** for predicting four
mechanical properties of silk fibers **directly from the protein sequence**:

> targets = [`toughness`, `E` (modulus), `strength`, `strain`]   →   metric = **mean of the 4 test R²**

This follows Karpathy's *autoresearch* loop: **propose → run → measure → ratchet**.

## Set up a run
Work in a **RUN clone** (see "Git workflow") on a per-session branch:
```bash
git checkout -b autoresearch/<tag>            # e.g. autoresearch/may30-silk
python setup.py --smoke-test                  # quick pipeline check (no full cache write)
python setup.py                               # one-time: cache embeddings (if cache/ missing)
git commit -am "baseline" 2>/dev/null; python run_experiment.py --tag baseline   # establish the bar
```

## The loop (repeat indefinitely) — git-ratcheted
**Commit the edit BEFORE running** so the commit hash run_experiment.py records points at *this*
experiment's code (this is what lets `export_experiments.py` / `analyze_results.py` recover it).
Every experiment is committed and **tagged**, so even rejected code stays recoverable.
1. **Read** `model.py` and `leaderboard.md` / `journal.md` (what's been tried — don't repeat dead ends).
2. **Form one hypothesis** likely to raise mean test R².
3. **Edit `model.py`** (and/or `config.json`). Keep the contract in `model.py`'s header.
4. **Commit, then tag:**
   ```bash
   git add model.py config.json && git commit -m "exp NNN: <idea>"
   git tag autoresearch-exp/<tag>-NNN          # preserves this snapshot even if reverted later
   ```
5. **Run:** `python run_experiment.py --tag "<idea>"` → prints `SCALAR mean test R² = …`, appends to
   `ledger.jsonl` (with this commit) + `leaderboard.md`, and stamps `status: keep/discard`.
6. **Ratchet:**
   - **Improved** over the best so far → keep: note why in `journal.md`, then
     `git add -A && git commit -m "keep exp NNN: <idea> R²=<value>"`.
   - **Did not improve** → revert the *code* to the last good architecture but keep the trace + lesson:
     ```bash
     git checkout <best-commit> -- model.py config.json   # best-commit from ledger.jsonl / git log
     ```
     append the **negative** result to `journal.md`, then
     `git add -A && git commit -m "reject exp NNN: <idea> R²=<value>"`. (The tag from step 4 keeps the
     rejected code recoverable; `ledger.jsonl` keeps its metrics so it isn't retried.)
7. Go to 1. One change per run. **Never stop** unless asked.

## Rules
- **Only edit `model.py` and `config.json`.** Do **not** modify `dataio.py`, `run_experiment.py`, the
  data, or the cache — the metric and splits must stay fixed and comparable.
- **No test peeking** beyond the single scalar the harness reports. The harness already early-stops on
  a *grouped* validation split (see "the catch"); do your model selection via that, not the test set.
- Keep each run reasonably fast (aim ≤ a few minutes) so you can iterate a lot. If an idea needs the
  backbone fine-tuned, see `baselines/lora_finetune.py` (slower, separate path).
- One change per run when possible, so you know what moved the metric.

## Git workflow (two clones: RUN vs EDIT)
Keep reusable infrastructure changes separate from optimization experiments by using two local clones:

```bash
# EDIT clone — change reusable code (dataio.py, run_experiment.py, baselines, docs) and push
git clone https://github.com/lamm-mit/EvolutionaryScale-protein-mechanics.git ESM-edit
# RUN clone — the agent experiments here; per-experiment commits accumulate
git clone https://github.com/lamm-mit/EvolutionaryScale-protein-mechanics.git ESM-run
```
- The **agent runs in the RUN clone** (`ESM-run/autoresearch_silk/`) and git-commits each experiment
  there (model.py + ledger/leaderboard/journal), as in the loop above.
- Reusable fixes (harness, new baselines, docs) are made in the **EDIT clone**, pushed, then
  `git pull`ed into RUN — so infra changes don't get tangled with experiment history.
- `data/` and `cache/` are git-ignored, so each clone runs `python setup.py` once (needs HF auth).
  The metric/splits are deterministic, so results stay comparable across clones.

## How the data/harness works
- `setup.py` cached **ESMC per-residue embeddings** for every sequence (so experiments are fast). Your
  model receives padded per-residue embeddings `x:(B,Lmax,d)` + `mask:(B,Lmax)` and outputs `(B,4)`.
- Targets are standardized for a balanced 4-task MSE; R² is computed on raw units (scale-invariant).
- **Train/test split** is controlled by `config.json`'s `split_mode`: `auto` (default — use the
  dataset's own `train`/`test` if it has them, else a harness split), `provided` (always the dataset's
  split), or `grouped` (pool all splits + dedup + a deterministic **leakage-safe** split grouped by
  property tuple, `test_frac`/`seed`, so identical-fiber sequences never straddle it). The internal
  early-stopping **validation** split is always grouped within train, regardless of mode.
- **Switch backbone** via `config.json`'s `esmc_model`; **switch dataset** via `dataset`
  (e.g. `lamm-mit/silkome-masp`). After any change re-run `python setup.py [--device cuda]`. Caches are
  keyed by (dataset, model) so they coexist.

## The catch (why this is hard — read this)
On the default **silkome-masp** set there are ~1028 distinct sequences but only ~233 distinct
measured-property tuples (silkome-full: ~3170 / ~268): **many sequences share one fiber measurement**
(properties are per-fiber/species; sequences are individual spidroins).
So the sequence→mechanics signal is weak and indirect. The validation split is **grouped by property
tuple** so identical-label fibers don't leak across train/val. Architectures that capture **repeat /
motif structure** (silk is highly repetitive: poly-A crystallites, GPGXX/GGX, GAGAGS) or that pool
the sequence cleverly are more likely to generalize than naive mean-pooling. Beware: it is easy to
"improve" by overfitting the small test set — prefer changes that also raise the grouped val R².

## Ideas (starting points — invent your own!)
Pooling (attention/gated/GeM/mean+max+std) · Conv1D / dilated convs over residues (motif detectors) ·
small Transformer / set-transformer · per-target heads vs shared trunk · log-transform skewed targets
(toughness, E) · uncertainty-weighted multi-task loss · concat composition/length/motif-count features ·
stronger backbone (600M/6B) · SAE features · LoRA fine-tuning (`baselines/`). Drop-in alternatives to
copy into `model.py` are in `baselines/`.

### Suggested research directions (with built-in support)

**1. Fusion / multi-task with taxonomy (predict species/genus/family/silk-type, then fuse).**
The data carries taxonomy columns (`family, genus, species, category1, category2, sex`). Train
**auxiliary classifiers from the sequence** jointly with the property head so they shape a shared
representation, then fuse. This is natively supported — a model may declare:
```python
AUX_COLS = ["family", "genus", "category1"]          # taxonomy targets predicted from sequence
def build_aux_heads(self, class_counts): ...          # run_experiment calls this with {col: n_classes}
def auxiliary_loss(self, x, mask, aux_labels): ...    # added to the loss during TRAINING only
```
See `baselines/fusion_taxonomy.py` (copy it over `model.py`); tune `aux_weight` in `config.json`.
Variants: two-stage (pretrain the taxonomy classifier, then fuse its frozen features), or use the
*predicted* taxonomy posterior as an extra input to the property head.
*Honest caveat:* a ground-truth `category1`-mean predictor scores R² ≈ 0 on this split, so taxonomy
is unlikely to be a silver bullet — its value is as a representation-shaping signal / regularizer.
It must be **predicted from sequence** (as here), never fed in as ground truth.

**2. Extract explicit sequence patterns (statistics or a GPT) to hone in on key subsequences.**
`dataio.sequence_motif_features(sequences)` gives per-sequence amino-acid composition + canonical
silk-motif counts (poly-A crystallites, `GPGGY`/`GPGQQ`, `GAGAGS`, `GGX`, poly-A run length) + length —
a ready building block. Sequences are in `data/*.parquet` (`SilkData.df["sequence"]`). Ideas: fuse
these stats with the embedding pool; or build features from a **protein GPT** (e.g. per-token
surprisal / hidden states from a generative model, à la SilkomeGPT) to flag the residues that carry
mechanical signal; or supervise **motif-presence as auxiliary targets** via the hook above. (Note:
Conv1D/attention over the per-residue embeddings is the in-harness way to learn motif detectors.)

**3. Other directions:** kNN-in-embedding-space baseline · per-family residual modeling (predict the
family mean + a sequence-conditioned residual) · ensembling across seeds/architectures · log-targets +
uncertainty-weighted 4-task loss · contrastive pretraining on sequence→property · and the big levers:
**LoRA fine-tuning** and a **stronger backbone (600M/6B)**.

## Reference measurements (calibration — read before optimizing)
Default dataset **silkome-masp** (ESMC-300M **mean-pooled** features, provided train/test split):

| approach | mean test R² |
|----------|----:|
| predict the global mean | ~0.00 (by definition) |
| **mean-pool → MLP (baseline)** | **~0.01** |
| Ridge (mean-pool features) | −0.41 … −0.02 (only ≈0 at very high regularization) |
| RandomForest (mean-pool) | −0.07 |
| `category1` (silk-type) mean predictor | −0.01 |

(silkome-full is the same story: baseline ≈ 0, Ridge/RF/category-mean ≤ 0, grouped-CV ≈ −0.3.)

**So the bar is brutal: nothing beats predicting the mean yet** — the single-sequence→fiber-mechanics
signal is extremely weak. **Any clearly positive mean test R² is a genuine result.** Don't be fooled
by tiny positive numbers (~0.01) — they're just "predict the mean" noise; aim for a robust, repeatable
gain that also shows up in grouped-val R².

**Most promising levers** (in rough order of expected payoff):
1. **LoRA fine-tuning** of ESMC end-to-end (`baselines/lora_finetune.py`) — lets the backbone
   specialize; the cached-embedding head search may be capped near 0.
2. **Stronger backbone**: `ESMC-600M` / `ESMC-6B` (edit `config.json` + re-run `setup.py`,
   ideally on a GPU). Bigger representations may expose signal mean-pool 300M can't.
3. **Per-residue sequence models** (Conv/attention/transformer) that read repeat/motif structure.
4. **Reframing**: per-target heads, log-targets, predicting family/category as an auxiliary task,
   or per-family residuals; robuster losses; ensembling.

Beat the top row of `leaderboard.md`. If after a real effort nothing clearly beats ~0, that
*itself* is a finding (fiber mechanics may need more than one spidroin sequence) — record it.
