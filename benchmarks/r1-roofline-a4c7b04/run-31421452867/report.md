# ArmBench MiniLM result

Generated: `2026-08-10T18:59:18.989010+00:00`

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
| 1 | 16 | 3.024 [2.996, 3.053] | 1.590 [1.585, 1.596] | 1.647 | 1.90x [1.88, 1.92] | 628.9 |
| 1 | 32 | 5.375 [5.339, 5.416] | 2.471 [2.462, 2.479] | 2.548 | 2.18x [2.16, 2.19] | 404.7 |
| 1 | 64 | 9.914 [9.888, 9.945] | 3.983 [3.969, 4.007] | 4.132 | 2.49x [2.47, 2.50] | 251.1 |
| 1 | 128 | 19.064 [19.020, 19.111] | 7.807 [7.733, 7.862] | 8.109 | 2.44x [2.42, 2.46] | 128.1 |
| 8 | 16 | 18.442 [18.399, 18.489] | 7.269 [7.235, 7.296] | 7.620 | 2.54x [2.53, 2.55] | 1100.6 |
| 8 | 32 | 33.858 [33.756, 33.914] | 11.535 [11.485, 11.567] | 11.916 | 2.94x [2.92, 2.95] | 693.5 |
| 8 | 64 | 67.484 [67.316, 67.692] | 22.631 [22.578, 22.674] | 23.657 | 2.98x [2.97, 2.99] | 353.5 |
| 8 | 128 | 136.617 [136.280, 137.016] | 47.799 [47.551, 48.134] | 51.081 | 2.86x [2.84, 2.88] | 167.4 |
| 32 | 16 | 66.698 [66.574, 66.911] | 21.447 [21.411, 21.487] | 22.273 | 3.11x [3.10, 3.12] | 1492.1 |
| 32 | 32 | 128.913 [128.554, 129.137] | 40.598 [40.478, 40.748] | 42.019 | 3.18x [3.16, 3.19] | 788.2 |
| 32 | 64 | 259.283 [258.840, 259.591] | 83.135 [82.959, 83.270] | 85.133 | 3.12x [3.11, 3.13] | 384.9 |
| 32 | 128 | 541.313 [540.505, 542.491] | 186.869 [186.228, 187.606] | 191.533 | 2.90x [2.88, 2.91] | 171.2 |

**Geometric-mean median-latency speedup: 2.69x.**

Maximum median-latency 95% CI half-width: **0.95%**. Tail-spike cases above 1.5x p95/median: **0**; likely VM preemption or host contention: **0**.

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
| baseline | `MatMul` | 8995369 | 960 |
| baseline | `BiasGelu` | 711777 | 120 |
| baseline | `LayerNormalization` | 523557 | 260 |
| baseline | `Softmax` | 302350 | 120 |
| baseline | `Add` | 158227 | 1000 |
| optimized | `DynamicQuantizeMatMul` | 1277921 | 360 |
| optimized | `Gelu` | 704622 | 120 |
| optimized | `MatMul` | 510665 | 240 |
| optimized | `MatMulIntegerToFloat` | 370282 | 360 |
| optimized | `Softmax` | 297935 | 120 |

## Interpretation

These measurements describe one pinned model, workload, runtime, and machine. They are not a universal Arm64 performance claim. Re-run the workflow on the target hardware before making deployment decisions.
