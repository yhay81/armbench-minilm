# R2 BF16 fast-math experiment

- Experiment ID: `r2-bf16-fastmath-v1`
- Parent: `r1-roofline-a4c7b04`
- Status: implemented; native Arm64 evidence pending
- Origin: agent-proposed after primary-source review

## Observation and hypothesis

The target Neoverse N2 runner advertises `bf16` and `svebf16`. The current
dynamic-QInt8 graph still executes 12 dynamic Attention MatMuls in floating
point, while the FP32 control spends most of its large-shape time in MatMul.
ONNX Runtime exposes the Arm64 MLAS session entry
`mlas.enable_gemm_fastmath_arm64_bfloat16=1` for accelerating FP32 GEMM through
BF16 instructions.

Hypothesis: enabling that one session entry reduces absolute latency for the
FP32 model and/or the residual floating-point GEMMs in the dynamic-QInt8 model.

Primary implementation sources:

- [ONNX Runtime session configuration keys](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)
- [Arm Neoverse N2 product page](https://developer.arm.com/Processors/Neoverse%20N2)

## Controlled variants

| Variant | Artifact | Session delta |
|---|---|---|
| `fp32_control` | pinned FP32 | none |
| `fp32_bf16_fastmath` | same pinned FP32 | BF16 fast-math enabled |
| `qint8_control` | current dynamic-QInt8 | none |
| `qint8_bf16_fastmath` | same dynamic-QInt8 | BF16 fast-math enabled |

All four variants use the same tokenizer revision, inputs, fixed shape grid,
four intra-op threads, sequential execution, spin settings, randomized blocks,
warm-ups, and bootstrap procedure. Tokenization remains outside the timing
boundary; mean pooling and L2 normalization remain inside it.

## Predeclared decision contract

Success signal:

- at least `1.03x` same-artifact geometric-mean median-latency speedup;
- no fixed-grid case below `0.97x`;
- mean embedding cosine against the FP32 control of at least `0.99`; and
- five independent native runs before promotion.

Follow-up signal: the aggregate speedup passes but one or more shapes regress by
over 3%; retain the candidate only for a shape-specific routing experiment.

Rejection signal: both same-artifact BF16 comparisons are at most `1.01x`, or a
candidate fails the quality gate. Abort when the target does not advertise BF16.

The authored 32-sentence fidelity check is a fast regression guard only. A
candidate cannot replace the submitted headline until the later pinned STS and
retrieval evaluation is also complete.

## Reproduction

```bash
uv run armbench-minilm prepare --work-dir .armbench
uv run armbench-minilm bf16-experiment \
  --work-dir .armbench \
  --output-dir results/experiments/r2-bf16-fastmath \
  --warmups 10 --block-warmups 3 --iterations 100 --measurement-blocks 5 \
  --sequence-lengths 16 32 64 128 --batch-sizes 1 8 32 --threads 4 \
  --bootstrap-resamples 2000 --random-seed 20260811 \
  --profile --profile-inferences 20
```

The command writes `experiment.json`, `experiment.md`, raw samples within each
measurement block, quality metrics, operator profiles, environment metadata,
artifact hashes, and an evidence-gated verdict. Generated result files remain
outside Git until a repeated native checkpoint is reviewed and promoted.
