# R2 BF16 fast-math checkpoint

This checkpoint preserves the first native Arm64 run of
`r2-bf16-fastmath-v1`. The implementation changes only the ONNX Runtime session
entry `mlas.enable_gemm_fastmath_arm64_bfloat16=1` while retaining the same
model artifacts, inputs, fixed shape grid, four threads, randomized blocks,
warm-ups, and timing boundary.

- Public code revision: [`a7ed33f`](https://github.com/yhay81/armbench-minilm/commit/a7ed33f5a80b51a8d342538101794a93d3a36baa)
- Native workflow: [run 31485130635](https://github.com/yhay81/armbench-minilm/actions/runs/31485130635)
- Runner: `ubuntu-24.04-arm`, Arm CPU part `0xd49`, BF16/SVEBF16 advertised
- Runtime: ONNX Runtime 1.28.0
- Protocol: 12 fixed shapes, 100 measured inferences per variant and shape, five randomized blocks, 2,000 bootstrap resamples, and 20 separately profiled inferences
- Verdict: `needs-independent-native-repeats`

## Result

| Same-artifact comparison | Geometric-mean speedup | Shape range | Quality versus FP32 control |
|---|---:|---:|---:|
| FP32 + BF16 fast-math vs FP32 | **1.3328x** | 1.3145x–1.3436x | 0.99998868 mean cosine |
| QInt8 + BF16 fast-math vs QInt8 | **1.0194x** | 1.0020x–1.0382x | 0.99271452 mean cosine |

The FP32 result passes the predeclared single-run effect, per-shape regression,
and quality gates. It is unusually consistent: all 12 shapes improve by more
than 31%. The QInt8 effect is smaller because most constant-weight MatMuls are
already integer kernels; its largest gain appears at batch 32, sequence 128.

At that largest shape:

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

The fastest observed candidate is QInt8 + BF16 fast-math: its geometric-mean
speedup over the FP32 control is 2.7534x on this fixed-shape run, compared with
2.7009x for QInt8 control. This does not replace the immutable submitted 2.45x
headline, which used an earlier protocol.

## Decision

Keep both BF16 variants for independent repetition. Do not promote either as a
new default yet: the predeclared contract requires five independent native runs,
and the authored-sentence cosine check does not replace the planned pinned STS
and retrieval evaluation.

The next independent hypothesis remains transformer optimization before
quantization, with exact `Attention`/`QAttention` fusion counts required before
any latency claim.

## Evidence

- [`run-31485130635/experiment.json`](run-31485130635/experiment.json) contains raw timed samples, bootstrap intervals, process-CPU samples, environment and model hashes, quality metrics, and all profile summaries.
- The generated Markdown ledger remains in the immutable GitHub Actions artifact `arm64-bf16-experiment-31485130635`; this README provides the tracked review summary.
- [`SHA256SUMS`](SHA256SUMS) authenticates the retained files.
