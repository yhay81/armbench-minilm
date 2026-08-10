# ArmBench MiniLM result

Generated: `2026-08-10T17:37:29.675547+00:00`

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
| 1 | 16 | 2.972 [2.948, 2.997] | 1.612 [1.607, 1.617] | 1.661 | 1.84x [1.83, 1.86] | 620.2 |
| 1 | 32 | 5.071 [5.043, 5.127] | 2.476 [2.468, 2.487] | 2.523 | 2.05x [2.03, 2.07] | 403.9 |
| 1 | 64 | 9.377 [9.346, 9.408] | 3.964 [3.956, 3.975] | 4.063 | 2.37x [2.36, 2.38] | 252.3 |
| 1 | 128 | 18.568 [18.540, 18.603] | 7.600 [7.582, 7.608] | 7.727 | 2.44x [2.44, 2.45] | 131.6 |
| 8 | 16 | 17.680 [17.637, 17.710] | 6.739 [6.724, 6.759] | 7.072 | 2.62x [2.61, 2.63] | 1187.1 |
| 8 | 32 | 33.596 [33.493, 33.672] | 11.350 [11.309, 11.373] | 14.301 | 2.96x [2.95, 2.97] | 704.8 |
| 8 | 64 | 67.012 [66.939, 67.081] | 22.116 [22.072, 22.206] | 26.043 | 3.03x [3.02, 3.04] | 361.7 |
| 8 | 128 | 135.640 [135.410, 135.929] | 47.431 [47.343, 47.519] | 49.565 | 2.86x [2.85, 2.87] | 168.7 |
| 32 | 16 | 65.850 [65.768, 66.004] | 20.982 [20.950, 21.000] | 21.248 | 3.14x [3.13, 3.15] | 1525.1 |
| 32 | 32 | 128.825 [128.658, 128.972] | 40.630 [40.592, 40.655] | 41.491 | 3.17x [3.17, 3.18] | 787.6 |
| 32 | 64 | 258.405 [258.203, 258.681] | 82.918 [82.845, 82.997] | 85.712 | 3.12x [3.11, 3.12] | 385.9 |
| 32 | 128 | 534.068 [533.666, 534.863] | 184.631 [184.433, 184.911] | 191.169 | 2.89x [2.89, 2.90] | 173.3 |

**Geometric-mean median-latency speedup: 2.67x.**

Maximum median-latency 95% CI half-width: **0.83%**. Tail-spike cases above 1.5x p95/median: **0**; likely VM preemption or host contention: **0**.

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
| baseline | `MatMul` | 1874061 | 192 |
| baseline | `BiasGelu` | 155037 | 24 |
| baseline | `LayerNormalization` | 105535 | 52 |
| baseline | `Softmax` | 60363 | 24 |
| baseline | `Add` | 35795 | 200 |
| optimized | `DynamicQuantizeMatMul` | 264514 | 72 |
| optimized | `Gelu` | 146426 | 24 |
| optimized | `MatMul` | 104578 | 48 |
| optimized | `MatMulIntegerToFloat` | 75825 | 72 |
| optimized | `Softmax` | 58994 | 24 |

## Interpretation

These measurements describe one pinned model, workload, runtime, and machine. They are not a universal Arm64 performance claim. Re-run the workflow on the target hardware before making deployment decisions.
