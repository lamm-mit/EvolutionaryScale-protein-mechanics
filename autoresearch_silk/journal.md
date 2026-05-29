# Research journal

Append-only log of hypotheses, experiments, and what was learned (especially **negative** results,
so they aren't repeated). Newest at the bottom.

---

- **(seed) baseline** — masked mean-pool → 2-layer MLP on ESMC-300M embeddings → mean test R² ≈ **0.01**
  (val −0.065). Attention-pool ≈ 0.00, Conv1D ≈ −0.02. All ≈ "predict the mean".
- **(calibration, classical)** — on the same split with mean-pool features: Ridge −0.12…−0.00,
  RandomForest −0.04, `category1`-mean −0.015, log-targets −0.05, grouped-CV Ridge −0.32. **173/175 test
  property-tuples are also in train, yet everything is ≤ 0.** Conclusion: with **mean-pooled ESMC-300M**
  the single-sequence→mechanics signal is essentially absent; floor is the global mean.
  → Next bets (untried): **LoRA fine-tuning**, **bigger backbone (600M/6B)**, **per-residue sequence
  models**, target reframing. The mean-pool head search alone is probably capped near 0.
