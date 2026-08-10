# R1 operator-scope ceiling checkpoint for `9aba1ab`

This checkpoint publishes the first reproducible Amdahl bound and repeated
memory-bandwidth inputs for ArmBench MiniLM. It is deliberately not labeled a
complete hardware roofline because independent exact-shape compute ceilings and
cache-aware traffic measurements are still missing.

## Immutable native Arm64 runs

- Commit: `9aba1ab611604678cd6f0a6a63658bce04a72546`
- CPU part: `0xd49`
- Runner image: `ubuntu24-arm64` version `20260719.67.1`
- ONNX Runtime: `1.28.0`, `CPUExecutionProvider`
- [Run 31417830941](https://github.com/yhay81/armbench-minilm/actions/runs/31417830941)
- [Run 31417840458](https://github.com/yhay81/armbench-minilm/actions/runs/31417840458)
- [Run 31417843628](https://github.com/yhay81/armbench-minilm/actions/runs/31417843628)

Each run used 100 retained timed samples per model/shape and a separate 20-call
profile per model/shape. The embedded profile nodes make this analysis
reproducible without the much larger raw ORT trace files.

## Exact operator scope

The analyzer matches ORT trace node names back to the ONNX graph and includes
only MatMul/Gemm nodes whose second input is a constant initializer. This finds
**36 quantization-target nodes** containing **40.500 MiB** of FP32 weights. The
12 dynamic Attention MatMuls are counted separately and are not incorrectly
treated as quantizable constant-weight operations.

Across the fixed grid:

- target nodes account for 56.98% to 81.54% of baseline profiled node time;
- the operator-scope infinite-speedup limit is **2.325x to 5.417x**;
- observed speedup reaches about 48.9% to 74.3% of the infinite improvement
  opportunity;
- the maximum run-to-run CV of the Amdahl limit is **1.723%**, down from 4.906%
  with four profiled inferences;
- profiled target work reaches up to 205.1 effective FP32 GFLOP/s and 1,026.1
  equivalent INT8 GOP/s. These are achieved rates, not independent compute
  ceilings.

The complete 12-shape table is in [`ceiling.md`](ceiling.md), and every formula
input and per-run result is retained in [`ceiling.json`](ceiling.json).

## Repeated memory-copy bandwidth

Effective bandwidth counts one source read and one destination write.

| Array size | Working set | Threads | Median of run medians | Run range | Run CV |
|---:|---:|---:|---:|---:|---:|
| 32 MiB | 64 MiB | 1 | 55.002 GB/s | 47.267–60.113 | 9.756% |
| 32 MiB | 64 MiB | 4 | 132.241 GB/s | 108.555–147.045 | 12.262% |
| 256 MiB | 512 MiB | 1 | 35.113 GB/s | 34.772–35.408 | 0.741% |
| 256 MiB | 512 MiB | 4 | **110.029 GB/s** | 107.369–113.477 | **2.267%** |

The preliminary projection uses the reproducible 256 MiB / four-thread median.
Moving the target's logical FP32 bytes at that copy rate would take 0.410 ms for
batch 1 / sequence 16 and 6.561 ms for batch 32 / sequence 128, versus 2.392 ms
and 424.308 ms of profiled target-node time in the first run. This comparison is
not a cache-aware bound and does not by itself prove whether a kernel is compute-
or bandwidth-limited.

## Promotion decision

No latency headline is promoted from these runs. The R1 operator metrics passed
their repeatability check, but batch 1 / sequence 16 baseline latency had 3.346%
run-to-run CV, and two batch 1 / sequence 32 runs contained a tail case. The
previous passing R0 validation remains the latency source of truth.

R1 remains open until exact-shape FP32 and dynamic-INT8 kernel microbenchmarks,
non-kernel timing separation, and a cache-aware traffic measurement complete the
hardware roofline.
