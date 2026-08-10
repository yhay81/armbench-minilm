# ArmBench operator-scope ceiling analysis

Exact scope: **36** constant-weight MatMul/Gemm nodes and 40.500 MiB of FP32 weights.

| Batch | Sequence | Target node-time share | Infinite Amdahl limit | Observed speedup | Potential realized |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 56.6% | 2.31x | 1.87x | 66.4% |
| 1 | 32 | 63.2% | 2.72x | 2.12x | 65.1% |
| 1 | 64 | 69.2% | 3.25x | 2.43x | 63.8% |
| 1 | 128 | 71.0% | 3.45x | 2.46x | 59.4% |
| 8 | 16 | 74.0% | 3.85x | 2.62x | 56.7% |
| 8 | 32 | 78.9% | 4.75x | 2.97x | 52.5% |
| 8 | 64 | 79.4% | 4.87x | 3.03x | 52.4% |
| 8 | 128 | 77.5% | 4.44x | 2.87x | 54.5% |
| 32 | 16 | 80.7% | 5.18x | 3.13x | 51.1% |
| 32 | 32 | 81.8% | 5.49x | 3.19x | 48.9% |
| 32 | 64 | 81.0% | 5.26x | 3.13x | 50.1% |
| 32 | 128 | 78.3% | 4.62x | 2.90x | 52.5% |

## Exact-shape kernel reference

Repeated runs: **3**; peak-case run-median CV: **0.86% FP32** and **0.77% QInt8**.

The maximum across every case is 32.17%; this includes sub-millisecond diagnostic cases and is not the CV of the peak used below.

| Batch | Sequence | Rows | QInt8 roofline minimum | Profiled QInt8 / roofline | Target-only Amdahl |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 16 | 0.302 ms | 3.42x | 1.99x |
| 1 | 32 | 32 | 0.603 ms | 2.50x | 2.17x |
| 1 | 64 | 64 | 1.206 ms | 1.91x | 2.38x |
| 1 | 128 | 128 | 2.413 ms | 1.60x | 2.45x |
| 8 | 16 | 128 | 2.413 ms | 1.59x | 2.59x |
| 8 | 32 | 256 | 4.825 ms | 1.33x | 2.86x |
| 8 | 64 | 512 | 9.651 ms | 1.20x | 2.92x |
| 8 | 128 | 1024 | 19.302 ms | 1.13x | 2.75x |
| 32 | 16 | 512 | 9.651 ms | 1.20x | 2.98x |
| 32 | 32 | 1024 | 19.302 ms | 1.13x | 3.04x |
| 32 | 64 | 2048 | 38.603 ms | 1.10x | 2.97x |
| 32 | 128 | 4096 | 77.206 ms | 1.10x | 2.78x |

Peak observed isolated rates: **207.4 FP32 GFLOP/s** and **1126.5 equivalent QInt8 GOP/s**.

The per-shape wall timings remain in `ceiling.json` as isolated-graph diagnostics. They are not multiplied into the limit because that would multiply the ORT invocation boundary once per target node.

## Roofline status

This is deliberately **preliminary**, not a completed hardware roofline.

Available:

- exact target and dynamic-attention MatMul FLOPs
- logical FP32 target bytes
- measured operator time
- optional measured copy bandwidth
- independent hot-weight FP32 and dynamic-QInt8 rates for every exact target shape

Still required:

- cache-hierarchy traffic or hardware-counter measurements

## Interpretation limits

- The infinite-speedup limit uses the target share of profiled ORT node time, so it is an operator-scope ceiling rather than a hardware peak claim.
- Logical bytes count each target weight, input, and output once per node; cache reuse and actual memory-controller traffic require separate measurement.
- Exact-shape kernels repeatedly reuse one synthetic weight per shape, making them an optimistic hot-weight reference rather than a physical hardware peak.
- The finite roofline Amdahl value changes only the profiled target time. It is a target-only projection, not an end-to-end ceiling when quantization also changes non-target work or runtime overhead.
- ORT profiling runs in separate sessions and does not alter timed latency samples.
