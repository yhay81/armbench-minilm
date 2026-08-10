# ArmBench MiniLM result

Generated: `2026-08-10T18:59:14.410942+00:00`

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
| 1 | 16 | 2.831 [2.819, 2.851] | 1.541 [1.537, 1.547] | 1.892 | 1.84x [1.83, 1.85] | 649.0 |
| 1 | 32 | 4.952 [4.940, 4.989] | 2.416 [2.412, 2.420] | 2.520 | 2.05x [2.04, 2.07] | 413.9 |
| 1 | 64 | 9.144 [9.123, 9.176] | 3.879 [3.870, 3.891] | 3.946 | 2.36x [2.35, 2.37] | 257.8 |
| 1 | 128 | 18.138 [18.080, 18.173] | 7.404 [7.394, 7.419] | 7.835 | 2.45x [2.44, 2.46] | 135.1 |
| 8 | 16 | 17.316 [17.268, 17.398] | 6.557 [6.542, 6.571] | 8.826 | 2.64x [2.63, 2.65] | 1220.2 |
| 8 | 32 | 32.867 [32.796, 32.976] | 10.962 [10.939, 10.975] | 12.251 | 3.00x [2.99, 3.01] | 729.8 |
| 8 | 64 | 66.202 [66.114, 66.284] | 21.610 [21.580, 21.639] | 22.230 | 3.06x [3.06, 3.07] | 370.2 |
| 8 | 128 | 134.920 [134.727, 135.096] | 46.574 [46.537, 46.623] | 48.667 | 2.90x [2.89, 2.90] | 171.8 |
| 32 | 16 | 65.000 [64.936, 65.112] | 20.545 [20.512, 20.582] | 21.557 | 3.16x [3.16, 3.17] | 1557.6 |
| 32 | 32 | 127.686 [127.576, 127.814] | 39.713 [39.667, 39.752] | 41.736 | 3.22x [3.21, 3.22] | 805.8 |
| 32 | 64 | 256.845 [256.516, 257.109] | 81.596 [81.537, 81.684] | 85.025 | 3.15x [3.14, 3.15] | 392.2 |
| 32 | 128 | 531.190 [530.406, 531.850] | 181.755 [181.546, 182.044] | 185.794 | 2.92x [2.92, 2.93] | 176.1 |

**Geometric-mean median-latency speedup: 2.69x.**

Maximum median-latency 95% CI half-width: **0.56%**. Tail-spike cases above 1.5x p95/median: **1**; likely VM preemption or host contention: **0**.

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
| baseline | `MatMul` | 8897573 | 960 |
| baseline | `BiasGelu` | 694081 | 120 |
| baseline | `LayerNormalization` | 517264 | 260 |
| baseline | `Softmax` | 302970 | 120 |
| baseline | `Add` | 134998 | 1000 |
| optimized | `DynamicQuantizeMatMul` | 1320976 | 360 |
| optimized | `Gelu` | 705001 | 120 |
| optimized | `MatMul` | 534509 | 240 |
| optimized | `MatMulIntegerToFloat` | 380949 | 360 |
| optimized | `Softmax` | 330193 | 120 |

## Interpretation

These measurements describe one pinned model, workload, runtime, and machine. They are not a universal Arm64 performance claim. Re-run the workflow on the target hardware before making deployment decisions.
