# ArmBench MiniLM result

Generated: `2026-08-10T18:58:57.255165+00:00`

## Environment

| Field | Value |
|---|---|
| Architecture | `aarch64` |
| OS | `Linux-6.17.0-1020-azure-aarch64-with-glibc2.39` |
| CPU | `aarch64` |
| CPU cores | 4 physical / 4 logical |
| ONNX Runtime | `1.28.0` |
| Python | `3.12.3` |
| Threads | 4 intra-op / 1 inter-op |
| Relevant Arm features | `asimd, i8mm, sve, sve2` |

## Measurement protocol

Fixed sequence lengths: `[16, 32, 64, 128]`; 5 balanced-randomized A/B blocks; 3 discarded warm-ups immediately before each model block; 100 raw samples per model and case.

Median intervals use 2000 deterministic bootstrap resamples. Wall-clock and process-CPU samples are both retained. Each timed sample is one inference; no outlier is discarded. Tokenization is excluded; timed samples include ONNX inference, mean pooling, and L2 normalization.

## Model size

| Model | Format | Size (MiB) | SHA-256 |
|---|---|---:|---|
| Baseline | FP32 | 86.22 | `6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452` |
| Optimized | QInt8 | 56.04 | `30265b36c31727ec2cfed302692feccd14a4272bb24c5af3e87a551b628bdabf` |

**Measured size reduction: 35.0%.**

## Inference performance

| Batch | Seq | FP32 median ms [95% CI] | INT8 median ms [95% CI] | INT8 p95 ms | Speedup [95% CI] | INT8 sentences/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 3.011 [2.971, 3.062] | 1.620 [1.609, 1.633] | 4.941 | 1.86x [1.83, 1.89] | 617.4 |
| 1 | 32 | 5.252 [5.228, 5.287] | 2.465 [2.459, 2.473] | 2.542 | 2.13x [2.12, 2.15] | 405.6 |
| 1 | 64 | 9.696 [9.669, 9.723] | 3.953 [3.935, 3.967] | 4.077 | 2.45x [2.44, 2.46] | 253.0 |
| 1 | 128 | 18.958 [18.934, 18.993] | 7.644 [7.618, 7.670] | 7.889 | 2.48x [2.47, 2.49] | 130.8 |
| 8 | 16 | 18.100 [18.083, 18.122] | 6.771 [6.760, 6.799] | 7.062 | 2.67x [2.66, 2.68] | 1181.4 |
| 8 | 32 | 33.655 [33.604, 33.691] | 11.326 [11.299, 11.368] | 11.726 | 2.97x [2.96, 2.98] | 706.4 |
| 8 | 64 | 67.036 [66.991, 67.155] | 22.121 [22.103, 22.157] | 23.687 | 3.03x [3.02, 3.04] | 361.6 |
| 8 | 128 | 136.040 [135.764, 136.269] | 47.410 [47.347, 47.481] | 50.049 | 2.87x [2.86, 2.88] | 168.7 |
| 32 | 16 | 65.919 [65.801, 66.027] | 21.065 [21.015, 21.085] | 21.918 | 3.13x [3.12, 3.14] | 1519.1 |
| 32 | 32 | 128.728 [128.552, 128.956] | 40.305 [40.265, 40.354] | 41.375 | 3.19x [3.19, 3.20] | 793.9 |
| 32 | 64 | 258.536 [258.146, 258.898] | 82.649 [82.548, 82.801] | 85.847 | 3.13x [3.12, 3.13] | 387.2 |
| 32 | 128 | 540.986 [540.329, 541.464] | 188.252 [187.985, 188.545] | 197.993 | 2.87x [2.87, 2.88] | 170.0 |

**Geometric-mean median-latency speedup: 2.70x.**

Maximum median-latency 95% CI half-width: **1.50%**. Tail-spike cases above 1.5x p95/median: **1**; likely VM preemption or host contention: **0**.

## Fidelity guardrail

Compared 32 authored sentences using normalized embeddings.

| Metric | Value |
|---|---:|
| Mean FP32/INT8 embedding cosine | 0.99267173 |
| Minimum FP32/INT8 embedding cosine | 0.97647780 |
| Mean pairwise-similarity absolute error | 0.01012334 |
| Maximum pairwise-similarity absolute error | 0.05169705 |

## Operator profile

Canonical profile: batch 32, sequence length 128.

| Model | Top operator | Profiled node time (µs) | Calls |
|---|---|---:|---:|
| baseline | `MatMul` | 8973144 | 960 |
| baseline | `BiasGelu` | 711235 | 120 |
| baseline | `LayerNormalization` | 520249 | 260 |
| baseline | `Softmax` | 302537 | 120 |
| baseline | `Add` | 160768 | 1000 |
| optimized | `DynamicQuantizeMatMul` | 1279424 | 360 |
| optimized | `Gelu` | 706670 | 120 |
| optimized | `MatMul` | 518310 | 240 |
| optimized | `MatMulIntegerToFloat` | 375253 | 360 |
| optimized | `Softmax` | 297992 | 120 |

## Interpretation

These measurements describe one pinned model, workload, runtime, and machine. They are not a universal Arm64 performance claim. Re-run the workflow on the target hardware before making deployment decisions.
