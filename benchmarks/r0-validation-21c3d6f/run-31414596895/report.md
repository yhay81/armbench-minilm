# ArmBench MiniLM result

Generated: `2026-08-10T17:37:31.244462+00:00`

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
| 1 | 16 | 2.840 [2.828, 2.857] | 1.535 [1.529, 1.541] | 1.612 | 1.85x [1.84, 1.87] | 651.3 |
| 1 | 32 | 5.001 [4.976, 5.042] | 2.430 [2.421, 2.441] | 2.525 | 2.06x [2.05, 2.08] | 411.5 |
| 1 | 64 | 9.382 [9.340, 9.405] | 3.883 [3.871, 3.891] | 4.020 | 2.42x [2.40, 2.43] | 257.6 |
| 1 | 128 | 18.324 [18.281, 18.385] | 7.476 [7.461, 7.487] | 7.845 | 2.45x [2.44, 2.46] | 133.8 |
| 8 | 16 | 17.602 [17.577, 17.638] | 6.622 [6.605, 6.639] | 7.964 | 2.66x [2.65, 2.67] | 1208.0 |
| 8 | 32 | 32.871 [32.839, 32.943] | 10.912 [10.895, 10.939] | 11.251 | 3.01x [3.00, 3.02] | 733.1 |
| 8 | 64 | 66.445 [66.331, 66.523] | 21.766 [21.723, 21.798] | 22.659 | 3.05x [3.05, 3.06] | 367.5 |
| 8 | 128 | 136.267 [136.092, 136.415] | 47.567 [47.473, 47.689] | 50.967 | 2.86x [2.86, 2.87] | 168.2 |
| 32 | 16 | 66.010 [65.899, 66.113] | 21.152 [21.124, 21.188] | 21.935 | 3.12x [3.11, 3.13] | 1512.9 |
| 32 | 32 | 129.394 [129.183, 129.761] | 40.723 [40.625, 40.794] | 41.970 | 3.18x [3.17, 3.19] | 785.8 |
| 32 | 64 | 258.305 [258.057, 258.518] | 82.634 [82.529, 82.822] | 84.917 | 3.13x [3.12, 3.13] | 387.2 |
| 32 | 128 | 533.578 [533.116, 534.344] | 183.906 [183.648, 184.143] | 188.151 | 2.90x [2.90, 2.91] | 174.0 |

**Geometric-mean median-latency speedup: 2.69x.**

Maximum median-latency 95% CI half-width: **0.66%**. Tail-spike cases above 1.5x p95/median: **0**; likely VM preemption or host contention: **0**.

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
| baseline | `MatMul` | 1777988 | 192 |
| baseline | `BiasGelu` | 140826 | 24 |
| baseline | `LayerNormalization` | 104212 | 52 |
| baseline | `Softmax` | 60370 | 24 |
| baseline | `Add` | 33159 | 200 |
| optimized | `DynamicQuantizeMatMul` | 251509 | 72 |
| optimized | `Gelu` | 140836 | 24 |
| optimized | `MatMul` | 104158 | 48 |
| optimized | `MatMulIntegerToFloat` | 73998 | 72 |
| optimized | `Softmax` | 58739 | 24 |

## Interpretation

These measurements describe one pinned model, workload, runtime, and machine. They are not a universal Arm64 performance claim. Re-run the workflow on the target hardware before making deployment decisions.
