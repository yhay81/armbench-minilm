# ArmBench MiniLM result

Generated: `2026-08-10T18:16:28.986118+00:00`

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
| 1 | 16 | 3.172 [3.164, 3.188] | 1.627 [1.621, 1.632] | 1.676 | 1.95x [1.94, 1.96] | 614.6 |
| 1 | 32 | 5.433 [5.418, 5.457] | 2.537 [2.530, 2.545] | 3.907 | 2.14x [2.13, 2.15] | 394.2 |
| 1 | 64 | 9.882 [9.842, 9.909] | 4.090 [4.072, 4.109] | 4.300 | 2.42x [2.40, 2.43] | 244.5 |
| 1 | 128 | 19.176 [19.143, 19.200] | 7.808 [7.789, 7.828] | 7.972 | 2.46x [2.45, 2.46] | 128.1 |
| 8 | 16 | 18.317 [18.290, 18.356] | 6.991 [6.970, 7.023] | 7.946 | 2.62x [2.61, 2.63] | 1144.4 |
| 8 | 32 | 34.127 [33.919, 34.388] | 11.448 [11.419, 11.492] | 13.309 | 2.98x [2.96, 3.01] | 698.8 |
| 8 | 64 | 67.229 [67.155, 67.337] | 22.428 [22.382, 22.459] | 23.708 | 3.00x [2.99, 3.01] | 356.7 |
| 8 | 128 | 136.296 [136.190, 136.503] | 47.836 [47.772, 47.905] | 52.815 | 2.85x [2.84, 2.85] | 167.2 |
| 32 | 16 | 66.273 [66.184, 66.392] | 21.384 [21.338, 21.406] | 22.147 | 3.10x [3.09, 3.11] | 1496.5 |
| 32 | 32 | 129.989 [129.885, 130.134] | 41.387 [41.321, 41.482] | 43.120 | 3.14x [3.13, 3.15] | 773.2 |
| 32 | 64 | 260.919 [260.627, 261.313] | 84.672 [84.526, 84.763] | 87.406 | 3.08x [3.08, 3.09] | 377.9 |
| 32 | 128 | 539.526 [538.792, 540.579] | 188.202 [187.989, 188.541] | 195.489 | 2.87x [2.86, 2.87] | 170.0 |

**Geometric-mean median-latency speedup: 2.69x.**

Maximum median-latency 95% CI half-width: **0.69%**. Tail-spike cases above 1.5x p95/median: **1**; likely VM preemption or host contention: **1**.

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
| baseline | `MatMul` | 8971374 | 960 |
| baseline | `BiasGelu` | 711334 | 120 |
| baseline | `LayerNormalization` | 521772 | 260 |
| baseline | `Softmax` | 301306 | 120 |
| baseline | `Add` | 164897 | 1000 |
| optimized | `DynamicQuantizeMatMul` | 1276791 | 360 |
| optimized | `Gelu` | 706116 | 120 |
| optimized | `MatMul` | 516951 | 240 |
| optimized | `MatMulIntegerToFloat` | 376464 | 360 |
| optimized | `Softmax` | 298129 | 120 |

## Interpretation

These measurements describe one pinned model, workload, runtime, and machine. They are not a universal Arm64 performance claim. Re-run the workflow on the target hardware before making deployment decisions.
