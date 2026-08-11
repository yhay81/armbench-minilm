# R2 BF16 fast-math checkpoint

This checkpoint preserves five independent native Arm64 runs of
`r2-bf16-fastmath-v1`. The implementation changes only the ONNX Runtime session
entry `mlas.enable_gemm_fastmath_arm64_bfloat16=1` while retaining the same
model artifacts, inputs, fixed shape grid, four threads, randomized blocks,
warm-ups, and timing boundary.

- Public code revision: [`a7ed33f`](https://github.com/yhay81/armbench-minilm/commit/a7ed33f5a80b51a8d342538101794a93d3a36baa)
- Repeated-run analyzer: [`feaa1f9`](https://github.com/yhay81/armbench-minilm/commit/feaa1f9cefc183f4b9c9a554a7c9235cec374549)
- Native workflows: [31485130635](https://github.com/yhay81/armbench-minilm/actions/runs/31485130635), [31486949258](https://github.com/yhay81/armbench-minilm/actions/runs/31486949258), [31486966714](https://github.com/yhay81/armbench-minilm/actions/runs/31486966714), [31486977343](https://github.com/yhay81/armbench-minilm/actions/runs/31486977343), and [31486986690](https://github.com/yhay81/armbench-minilm/actions/runs/31486986690)
- Runner: `ubuntu-24.04-arm`, Arm CPU part `0xd49`, BF16/SVEBF16 advertised
- Runtime: ONNX Runtime 1.28.0
- Protocol: 12 fixed shapes, 100 measured inferences per variant and shape, five randomized blocks, 2,000 bootstrap resamples, and 20 separately profiled inferences
- Verdict: `performance-repetition-gate-passed-needs-task-quality`

## Result

| Same-artifact comparison | Median run GM | Run-GM CV | All-run/shape range | Quality versus FP32 control |
|---|---:|---:|---:|---:|
| FP32 + BF16 fast-math vs FP32 | **1.3348x** | **0.38%** | 1.3001x–1.3707x | 0.99998868 mean cosine |
| QInt8 + BF16 fast-math vs QInt8 | **1.0194x** | **0.18%** | 0.9909x–1.0507x | 0.99271452 mean cosine |

FP32 passes the predeclared five-run performance, per-shape regression,
stability, and authored-sentence fidelity gates. Every run has a 1.3248x–1.3386x
geometric-mean improvement; the maximum shape-level run CV is 1.96%. QInt8 does
not pass the 1.03x universal-effect threshold because most constant-weight
MatMuls are already integer kernels. Its stable gains at sequence 128 justify a
shape-specific follow-up rather than universal adoption.

The first run's absolute latencies at the largest shape were:

| Variant | Median latency |
|---|---:|
| FP32 control | 532.704 ms |
| FP32 + BF16 fast-math | 405.259 ms |
| QInt8 control | 183.412 ms |
| QInt8 + BF16 fast-math | **176.670 ms** |

The separately collected 20-inference profile attributes the change to the
expected operator. FP32 MatMul time falls from 8,903,793 µs to 6,324,807 µs in
aggregate (about 1.41x), while the residual QInt8-graph MatMul time falls from
512,305 µs to 373,706 µs (about 1.37x). DynamicQuantizeMatMul and the nonlinear
operators remain essentially unchanged.

The fastest first-run candidate is QInt8 + BF16 fast-math: its geometric-mean
speedup over the FP32 control is 2.7534x, compared with 2.7009x for QInt8
control. This does not replace the immutable submitted 2.45x headline, which
used an earlier protocol.

## Decision

Retain FP32 + BF16 for the pinned STS and retrieval quality gate. Its native
performance repetition gate is complete. Retain QInt8 + BF16 only as a
shape-specific candidate, led by sequence-128 workloads; its 1.0194x universal
effect is below the predeclared materiality threshold.

The first run used revision `a7ed33f`; the remaining four used `da709ab`. A
revision diff confirms that only documentation, the manifest, and retained
benchmark evidence changed. Runtime source, tests, workflow, lockfile, and model
artifacts are identical.

The next independent hypothesis remains transformer optimization before
quantization, with exact `Attention`/`QAttention` fusion counts required before
any latency claim.

## Evidence

- [`aggregate.md`](aggregate.md) and [`aggregate.json`](aggregate.json) contain the reproducible five-run decision and shape-level stability analysis.
- Each `run-*/experiment.json` contains raw timed samples, bootstrap intervals, process-CPU samples, environment and model hashes, quality metrics, and all profile summaries.
- Generated Markdown ledgers remain in the corresponding immutable GitHub Actions artifacts; this README provides the tracked review summary.
- [`SHA256SUMS`](SHA256SUMS) authenticates the retained files.
