# ArmBench MiniLM result

Generated: `2026-08-10T17:37:05.541517+00:00`

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
| 1 | 16 | 2.998 [2.963, 3.019] | 1.635 [1.629, 1.639] | 1.705 | 1.83x [1.81, 1.85] | 611.8 |
| 1 | 32 | 5.107 [5.090, 5.138] | 2.508 [2.496, 2.520] | 2.582 | 2.04x [2.03, 2.05] | 398.7 |
| 1 | 64 | 9.347 [9.307, 9.402] | 3.968 [3.958, 3.977] | 4.051 | 2.36x [2.34, 2.37] | 252.0 |
| 1 | 128 | 18.508 [18.420, 18.567] | 7.570 [7.555, 7.582] | 7.748 | 2.44x [2.43, 2.45] | 132.1 |
| 8 | 16 | 17.568 [17.543, 17.608] | 6.731 [6.717, 6.737] | 6.908 | 2.61x [2.61, 2.62] | 1188.6 |
| 8 | 32 | 33.144 [33.028, 33.220] | 11.141 [11.125, 11.170] | 11.471 | 2.97x [2.96, 2.98] | 718.1 |
| 8 | 64 | 66.967 [66.823, 67.081] | 22.040 [21.956, 22.102] | 26.281 | 3.04x [3.03, 3.05] | 363.0 |
| 8 | 128 | 135.742 [135.554, 135.973] | 47.182 [47.113, 47.256] | 48.159 | 2.88x [2.87, 2.88] | 169.6 |
| 32 | 16 | 65.663 [65.589, 65.800] | 20.937 [20.912, 20.998] | 21.667 | 3.14x [3.13, 3.14] | 1528.4 |
| 32 | 32 | 128.713 [128.580, 129.026] | 40.359 [40.322, 40.447] | 41.911 | 3.19x [3.18, 3.20] | 792.9 |
| 32 | 64 | 258.299 [258.088, 258.579] | 82.639 [82.550, 82.758] | 85.621 | 3.13x [3.12, 3.13] | 387.2 |
| 32 | 128 | 533.998 [533.640, 534.693] | 183.546 [183.149, 183.759] | 188.230 | 2.91x [2.91, 2.92] | 174.3 |

**Geometric-mean median-latency speedup: 2.67x.**

Maximum median-latency 95% CI half-width: **0.94%**. Tail-spike cases above 1.5x p95/median: **0**; likely VM preemption or host contention: **0**.

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
| baseline | `MatMul` | 1786482 | 192 |
| baseline | `BiasGelu` | 140887 | 24 |
| baseline | `LayerNormalization` | 102704 | 52 |
| baseline | `Softmax` | 60138 | 24 |
| baseline | `Add` | 32782 | 200 |
| optimized | `DynamicQuantizeMatMul` | 257718 | 72 |
| optimized | `Gelu` | 136125 | 24 |
| optimized | `MatMul` | 104616 | 48 |
| optimized | `MatMulIntegerToFloat` | 77184 | 72 |
| optimized | `Softmax` | 59149 | 24 |

## Interpretation

These measurements describe one pinned model, workload, runtime, and machine. They are not a universal Arm64 performance claim. Re-run the workflow on the target hardware before making deployment decisions.
