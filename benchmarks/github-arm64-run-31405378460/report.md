# ArmBench MiniLM result

Generated: `2026-08-10T15:46:37.106847+00:00`

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

## Model size

| Model | Format | Size (MiB) | SHA-256 |
|---|---|---:|---|
| Baseline | FP32 | 86.22 | `6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452` |
| Optimized | QInt8 | 56.04 | `30265b36c31727ec2cfed302692feccd14a4272bb24c5af3e87a551b628bdabf` |

**Measured size reduction: 35.0%.**

## Inference performance

Tokenization is excluded. Each timed sample includes ONNX inference, mean pooling, and L2 normalization.

| Batch | FP32 median (ms) | INT8 median (ms) | INT8 p95 (ms) | Speedup | INT8 sentences/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.928 | 1.629 | 1.665 | 1.80x | 613.7 |
| 8 | 19.331 | 7.376 | 14.299 | 2.62x | 1084.6 |
| 32 | 74.264 | 23.815 | 25.158 | 3.12x | 1343.7 |

**Geometric-mean median-latency speedup: 2.45x.**

## Fidelity guardrail

Compared 32 authored sentences using normalized embeddings.

| Metric | Value |
|---|---:|
| Mean FP32/INT8 embedding cosine | 0.99267173 |
| Minimum FP32/INT8 embedding cosine | 0.97647780 |
| Mean pairwise-similarity absolute error | 0.01012334 |
| Maximum pairwise-similarity absolute error | 0.05169705 |

## Interpretation

These measurements describe one pinned model, workload, runtime, and machine. They are not a universal Arm64 performance claim. Re-run the workflow on the target hardware before making deployment decisions.
