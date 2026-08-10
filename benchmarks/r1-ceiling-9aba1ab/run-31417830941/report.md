# ArmBench MiniLM result

Generated: `2026-08-10T18:16:16.981317+00:00`

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
| 1 | 16 | 3.174 [3.159, 3.189] | 1.645 [1.639, 1.653] | 1.812 | 1.93x [1.92, 1.94] | 607.9 |
| 1 | 32 | 5.581 [5.531, 5.608] | 2.582 [2.564, 2.608] | 6.851 | 2.16x [2.13, 2.18] | 387.2 |
| 1 | 64 | 9.870 [9.851, 9.892] | 4.052 [4.047, 4.077] | 4.225 | 2.44x [2.42, 2.44] | 246.8 |
| 1 | 128 | 19.106 [19.075, 19.153] | 7.806 [7.788, 7.824] | 7.992 | 2.45x [2.44, 2.46] | 128.1 |
| 8 | 16 | 18.220 [18.204, 18.238] | 6.930 [6.916, 6.949] | 7.137 | 2.63x [2.62, 2.63] | 1154.4 |
| 8 | 32 | 33.809 [33.777, 33.851] | 11.530 [11.506, 11.552] | 11.736 | 2.93x [2.93, 2.94] | 693.8 |
| 8 | 64 | 67.482 [67.419, 67.734] | 22.570 [22.539, 22.597] | 23.792 | 2.99x [2.98, 3.00] | 354.5 |
| 8 | 128 | 136.731 [136.490, 136.999] | 47.857 [47.815, 47.907] | 50.073 | 2.86x [2.85, 2.86] | 167.2 |
| 32 | 16 | 66.147 [66.099, 66.227] | 21.232 [21.193, 21.264] | 22.302 | 3.12x [3.11, 3.12] | 1507.2 |
| 32 | 32 | 129.108 [129.006, 129.396] | 40.774 [40.705, 40.822] | 46.761 | 3.17x [3.16, 3.17] | 784.8 |
| 32 | 64 | 259.443 [259.172, 259.719] | 83.524 [83.407, 83.730] | 89.963 | 3.11x [3.10, 3.11] | 383.1 |
| 32 | 128 | 541.591 [541.121, 542.341] | 189.910 [189.675, 190.251] | 195.193 | 2.85x [2.85, 2.86] | 168.5 |

**Geometric-mean median-latency speedup: 2.69x.**

Maximum median-latency 95% CI half-width: **0.86%**. Tail-spike cases above 1.5x p95/median: **1**; likely VM preemption or host contention: **0**.

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
| baseline | `MatMul` | 9012741 | 960 |
| baseline | `BiasGelu` | 713646 | 120 |
| baseline | `LayerNormalization` | 528857 | 260 |
| baseline | `Softmax` | 300533 | 120 |
| baseline | `Add` | 170928 | 1000 |
| optimized | `DynamicQuantizeMatMul` | 1286639 | 360 |
| optimized | `Gelu` | 723557 | 120 |
| optimized | `MatMul` | 513866 | 240 |
| optimized | `MatMulIntegerToFloat` | 375165 | 360 |
| optimized | `Softmax` | 295308 | 120 |

## Interpretation

These measurements describe one pinned model, workload, runtime, and machine. They are not a universal Arm64 performance claim. Re-run the workflow on the target hardware before making deployment decisions.
