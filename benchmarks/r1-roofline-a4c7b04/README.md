# R1 exact-shape compute-roofline checkpoint

Status: accepted as the independent compute-rate checkpoint for R1. The hardware
roofline remains preliminary until cache-hierarchy traffic or hardware counters
replace minimum logical-byte accounting.

This evidence measures every constant-weight MiniLM MatMul shape on the same
native Arm64 runner class as the full-model profiles, then connects the best
sustainable exact-shape rate to the profiled target pipeline.

## Provenance

- Public source commit: `a4c7b0448ae3976bd94d43f2f5cff8734cef96df`
- Workflow runs: [31421425620](https://github.com/yhay81/armbench-minilm/actions/runs/31421425620), [31421449317](https://github.com/yhay81/armbench-minilm/actions/runs/31421449317), and [31421452867](https://github.com/yhay81/armbench-minilm/actions/runs/31421452867)
- Runner: GitHub `ubuntu-24.04-arm`, Arm CPU part `0xd49`, runner image `20260719.67.1`
- Baseline model SHA-256: recorded and verified against every full-model and kernel result

An earlier implementation run, `31420246017` at `959b880`, is deliberately
excluded. It multiplied isolated graph invocation time by 36 and therefore
multiplied the Python/ORT invocation boundary once per target node. The raw
isolated timings remain useful diagnostics, but the corrected roofline uses the
best measured exact-shape compute rate and does not repeat that boundary.

## Measurement contract

- Exact target inventory: `(384 x 384) x 24`, `(384 x 1536) x 6`, and `(1536 x 384) x 6`
- Row counts: `16, 32, 64, 128, 256, 512, 1024, 2048, 4096`
- Total cases: 27 shapes; both FP32 and dynamic per-channel signed QInt8
- Per case: 100 retained single-inference samples in five balanced randomized A/B blocks
- Per block: three discarded warm-ups per precision
- Runtime: four intra-op threads, sequential execution, all graph optimizations, pinned spin duration/backoff
- Repeats: three independent native Arm64 workflows at the same source commit

The generated one-MatMul graphs use deterministic synthetic weights. QInt8 uses
the same `quantize_dynamic` settings as the full model. Repeated calls are an
optimistic hot-weight reference, not a physical processor peak.

## Results

| Quantity | Three-run result |
|---|---:|
| Best FP32 exact-shape rate | **207.431 GFLOP/s** |
| FP32 peak-case run-median CV | **0.863%** |
| Best QInt8 exact-shape rate | **1,126.502 equivalent GOP/s** |
| QInt8 peak-case run-median CV | **0.768%** |
| Profiled QInt8 target time / roofline minimum | **1.10x–3.42x** |
| Target-only Amdahl projection at the roofline | **1.99x–3.04x** |

The ceiling comes from stable large cases: FP32 uses rows 4096 at `1536 x 384`,
and QInt8 uses rows 4096 at `384 x 1536`. The maximum run-median CV over all 27
diagnostic cases is 32.168%, but it occurs in a sub-0.2 ms case and is not used
as the peak. Reporting both prevents the stable ceiling cases from hiding timer-
scale variability in the smallest kernels.

The preliminary QInt8 roofline is compute-dominated in all 36 combinations of
three runs and 12 full-model workloads. Minimum logical traffic reaches at most
35.98% of the compute-time projection. This does not prove actual cache or DRAM
traffic; dynamic-quantization scratch traffic and cache reuse remain unmeasured.

The largest remaining target-pipeline gap is at batch 1, sequence 16: the
profiled QInt8 target takes 3.42 times its 0.302 ms roofline minimum. The gap
shrinks to about 1.10x at the largest workloads. This points R2 toward invocation,
allocation, activation-quantization, and graph overhead at small shapes rather
than more work on large MatMul throughput.

The finite Amdahl value changes only the profiled target time. It is not a strict
end-to-end ceiling because quantization can also change non-target work and
runtime overhead. The separate infinite target-operator limit remains the upper
operator-scope reference.

## Promotion decision

The exact-shape compute rates are accepted: both peak-case run CVs are below 1%.
The full-model latency results in these runs are not promoted over the passing R0
evidence. Their maximum median-CI half-width was 1.50%, two runs contained an
in-process p95 spike, and the maximum workload speedup CV was 2.46%. Profiling
and kernel measurement use separate sessions, so this does not invalidate the
compute checkpoint.

## Evidence layout

- `ceiling.json` and `ceiling.md`: aggregate three-run Amdahl/roofline analysis
- `run-*/kernel-ceiling.json`: raw block orders and all exact-shape samples
- `run-*/benchmark.json`: full-model samples plus embedded 20-inference ORT node summaries
- `run-*/memory-bandwidth.json`: native copy-bandwidth samples
- `run-*/report.md`: generated full-model report for that run
- `SHA256SUMS`: integrity manifest for every retained file

Raw ORT trace files are omitted because the necessary node names, shapes, call
counts, and durations are embedded in each `benchmark.json`. No credentials,
local paths, generated model binaries, or private inputs are included.

## Remaining R1 work

The exact-shape compute-rate gap is now measured. Completing a defensible
hardware roofline still requires cache-aware traffic or hardware-counter data.
Until that exists, the reported memory term is explicitly a minimum logical-
traffic projection.
