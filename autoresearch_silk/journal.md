# Research journal

Append-only log of hypotheses, experiments, and what was learned (especially **negative** results,
so they aren't repeated). Newest at the bottom.

---

- **(default dataset = `silkome-masp`, provided train/test split; ESMC-300M)** — baseline masked
  mean-pool → MLP → mean test R² ≈ **0.006** (val −0.023). Classical (mean-pool features): Ridge
  −0.41…−0.02, RandomForest −0.07, `category1`-mean −0.01. Same story as silkome-full: nothing beats
  the mean yet. (Ledger was reset for the dataset switch; this is the bar to beat.)

### Earlier findings on `silkome-full` (for reference)
- **(seed) baseline** — masked mean-pool → 2-layer MLP on ESMC-300M embeddings → mean test R² ≈ **0.01**
  (val −0.065). Attention-pool ≈ 0.00, Conv1D ≈ −0.02. All ≈ "predict the mean".
- **(calibration, classical)** — on the same split with mean-pool features: Ridge −0.12…−0.00,
  RandomForest −0.04, `category1`-mean −0.015, log-targets −0.05, grouped-CV Ridge −0.32. **173/175 test
  property-tuples are also in train, yet everything is ≤ 0.** Conclusion: with **mean-pooled ESMC-300M**
  the single-sequence→mechanics signal is essentially absent; floor is the global mean.
  → Next bets (untried): **LoRA fine-tuning**, **bigger backbone (600M/6B)**, **per-residue sequence
  models**, target reframing. The mean-pool head search alone is probably capped near 0.
- **(GPU runs, NVIDIA GB10 / DGX Spark)** — reproduced on CUDA: mean-pool baseline mean test R² ≈
  **−0.03**; **300M LoRA** end-to-end (1.85M trainable, 3 epochs) → loss flat ~180, mean test R²
  ≈ **−0.001** (still "predict the mean"). **ESMC-6B + LoRA** (batch 4, max-len 512, bf16) started at
  epoch-1 val **−0.126** but was **~1 hr/epoch** — killed as too slow for an interactive trial; left
  for a future agent to explore with faster settings (bigger batch, shorter max-len, torch.compile,
  fewer steps). Takeaway so far: nothing beats the mean yet; the open levers remain 6B/LoRA at scale,
  per-residue sequence models, and reframing the target/task.
