# ArmBench operator-scope ceiling analysis

Exact scope: **36** constant-weight MatMul/Gemm nodes and 40.500 MiB of FP32 weights.

| Batch | Sequence | Target node-time share | Infinite Amdahl limit | Observed speedup | Potential realized |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 57.0% | 2.33x | 1.98x | 74.3% |
| 1 | 32 | 63.7% | 2.76x | 2.19x | 67.4% |
| 1 | 64 | 69.3% | 3.26x | 2.43x | 63.4% |
| 1 | 128 | 70.8% | 3.43x | 2.43x | 59.1% |
| 8 | 16 | 73.8% | 3.82x | 2.62x | 57.4% |
| 8 | 32 | 79.0% | 4.76x | 2.95x | 51.9% |
| 8 | 64 | 79.1% | 4.79x | 3.00x | 52.7% |
| 8 | 128 | 77.3% | 4.40x | 2.85x | 54.5% |
| 32 | 16 | 80.4% | 5.11x | 3.10x | 51.2% |
| 32 | 32 | 81.5% | 5.42x | 3.16x | 48.9% |
| 32 | 64 | 80.8% | 5.21x | 3.09x | 49.7% |
| 32 | 128 | 78.1% | 4.56x | 2.86x | 52.2% |

## Roofline status

This is deliberately **preliminary**, not a completed hardware roofline.

Available:

- exact target and dynamic-attention MatMul FLOPs
- logical FP32 target bytes
- measured operator time
- optional measured copy bandwidth

Still required:

- independent FP32 and INT8 compute ceilings for the exact matrix shapes
- cache-hierarchy traffic or hardware-counter measurements

## Interpretation limits

- The infinite-speedup limit uses the target share of profiled ORT node time, so it is an operator-scope ceiling rather than a hardware peak claim.
- Logical bytes count each target weight, input, and output once per node; cache reuse and actual memory-controller traffic require separate measurement.
- ORT profiling runs in separate sessions and does not alter timed latency samples.
