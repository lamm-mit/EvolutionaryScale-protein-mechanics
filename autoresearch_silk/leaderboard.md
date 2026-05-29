# Leaderboard — mean test R² (silk mechanics from sequence)

_3 experiments logged. Higher is better; baseline-to-beat is the top row._

| rank | mean R² | toughness | E | strength | strain | model | backbone | tag |
|----:|----:|----:|----:|----:|----:|------|------|-----|
| 1 | **0.010** | 0.017 | -0.001 | -0.006 | 0.030 | baseline: masked mean-pool -> 2-layer MLP | ESMC-300M | baseline |
| 2 | **-0.001** | -0.005 | -0.006 | -0.000 | 0.007 | attention pooling (learned query) -> MLP | ESMC-300M | attention pooling |
| 3 | **-0.020** | -0.060 | -0.011 | 0.008 | -0.018 | multi-scale Conv1D motif detectors -> mean+max p | ESMC-300M | conv1d motifs |
