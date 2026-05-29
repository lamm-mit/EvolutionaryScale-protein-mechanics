# Baselines / starting points

Drop-in alternatives for `../model.py` (same `build_model(embed_dim, n_targets, cfg)` contract).
Copy one over `model.py`, then `python run_experiment.py --tag "..."`:

```bash
cp baselines/attention_pool.py model.py && python run_experiment.py --tag "attention pooling"
cp baselines/conv1d.py        model.py && python run_experiment.py --tag "conv1d motifs"
```

- **`meanpool_mlp.py`** — the seeded baseline (reference copy).
- **`attention_pool.py`** — learned attention pooling over residues (vs naive mean).
- **`conv1d.py`** — multi-scale Conv1D motif detectors over the residue embeddings (silk is repetitive),
  then masked mean+max pooling.

### Harder path — LoRA fine-tuning of the backbone
`lora_finetune.py` is a **separate, self-contained** experiment that LoRA-fine-tunes ESMC end-to-end
(not on cached embeddings; slower, needs `peft`). It reads `data/{train,test}.parquet`, trains a
4-target regression head with LoRA adapters, and prints mean test R².
```bash
python baselines/lora_finetune.py --epochs 3 --lora-r 16 --device cuda
```
These are *starting points* — the goal is to invent architectures that beat them on mean test R².
