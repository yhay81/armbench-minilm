# ArmBench MiniLM result

Generated: `2026-08-10T18:16:36.445477+00:00`

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
| 1 | 16 | 3.404 [3.360, 3.417] | 1.640 [1.633, 1.646] | 2.068 | 2.08x [2.05, 2.09] | 609.7 |
| 1 | 32 | 5.787 [5.765, 5.814] | 2.567 [2.559, 2.578] | 2.712 | 2.25x [2.24, 2.27] | 389.6 |
| 1 | 64 | 10.216 [10.201, 10.236] | 4.165 [4.153, 4.180] | 4.309 | 2.45x [2.44, 2.46] | 240.1 |
| 1 | 128 | 19.291 [19.252, 19.320] | 8.056 [8.035, 8.084] | 8.294 | 2.39x [2.38, 2.40] | 124.1 |
| 8 | 16 | 18.517 [18.483, 18.544] | 7.102 [7.080, 7.127] | 7.277 | 2.61x [2.60, 2.62] | 1126.5 |
| 8 | 32 | 33.946 [33.913, 33.973] | 11.556 [11.535, 11.577] | 11.774 | 2.94x [2.93, 2.94] | 692.3 |
| 8 | 64 | 67.974 [67.873, 68.177] | 22.676 [22.631, 22.729] | 23.068 | 3.00x [2.99, 3.01] | 352.8 |
| 8 | 128 | 137.020 [136.742, 137.284] | 48.083 [48.027, 48.104] | 50.770 | 2.85x [2.84, 2.86] | 166.4 |
| 32 | 16 | 66.634 [66.556, 66.701] | 21.559 [21.506, 21.598] | 22.405 | 3.09x [3.08, 3.10] | 1484.3 |
| 32 | 32 | 129.918 [129.753, 130.140] | 41.034 [41.006, 41.087] | 41.835 | 3.17x [3.16, 3.17] | 779.8 |
| 32 | 64 | 260.584 [260.348, 261.287] | 84.299 [84.219, 84.461] | 88.323 | 3.09x [3.08, 3.10] | 379.6 |
| 32 | 128 | 540.793 [539.547, 541.621] | 189.081 [188.817, 189.572] | 193.739 | 2.86x [2.85, 2.87] | 169.2 |

**Geometric-mean median-latency speedup: 2.71x.**

Maximum median-latency 95% CI half-width: **0.84%**. Tail-spike cases above 1.5x p95/median: **0**; likely VM preemption or host contention: **0**.

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
| baseline | `MatMul` | 9024501 | 960 |
| baseline | `BiasGelu` | 720836 | 120 |
| baseline | `LayerNormalization` | 522219 | 260 |
| baseline | `Softmax` | 306254 | 120 |
| baseline | `Add` | 171813 | 1000 |
| optimized | `DynamicQuantizeMatMul` | 1286207 | 360 |
| optimized | `Gelu` | 724378 | 120 |
| optimized | `MatMul` | 518153 | 240 |
| optimized | `MatMulIntegerToFloat` | 378076 | 360 |
| optimized | `Softmax` | 299083 | 120 |

## Interpretation

These measurements describe one pinned model, workload, runtime, and machine. They are not a universal Arm64 performance claim. Re-run the workflow on the target hardware before making deployment decisions.
