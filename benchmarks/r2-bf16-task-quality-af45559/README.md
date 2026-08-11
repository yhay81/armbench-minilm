# R2 BF16 downstream task-quality checkpoint

This directory retains the first native Arm64 execution of the predeclared
[`r2-bf16-task-quality-v1`](../../docs/experiments/r2-bf16-task-quality-v1.md)
contract.

- Source commit: `af45559851b8a500c02df8155a44bd8b8f76b9a1`
- Workflow run: [31495451767](https://github.com/yhay81/armbench-minilm/actions/runs/31495451767)
- Job: `bf16-task-quality-gate`
- Artifact: `arm64-bf16-task-quality-31495451767`
- Runner: native Arm64, Neoverse N2 (`0xd49`), four CPU cores, ONNX Runtime 1.28.0

## Result

| Comparison | STS loss | ArguAna relative loss | STS cosine | Retrieval cosine | Verdict |
|---|---:|---:|---:|---:|---|
| FP32+BF16 vs FP32 | -0.0096 points | -0.0856% | 0.99998246 | 0.99998768 | **Pass** |
| QInt8+BF16 vs QInt8 | 0.3059 points | 0.5666% | 0.99025232 | 0.99515090 | **Pass** |
| QInt8 vs FP32 | -0.7890 points | -0.0265% | 0.98911148 | 0.99431392 | **Reject: STS cosine** |
| QInt8+BF16 vs FP32 | -0.4831 points | 0.5403% | 0.98910185 | 0.99432262 | **Reject: STS cosine** |

Negative loss means the candidate score was slightly higher. The retained FP32+BF16 option has
now passed both the separate five-run performance gate (1.3348x median geometric-mean speedup,
0.38% run CV) and this task-quality gate. QInt8+BF16 preserved the QInt8 artifact's task quality,
but its 1.0194x performance result remains shape-specific and is not a new headline.

The submitted QInt8 artifact did not lose either task score relative to FP32, but its 0.98911148
corresponding-embedding cosine on the STS inputs missed the deliberately strict 0.99 gate. The
historical 2.45x submission result remains immutable and is reported with this limitation rather
than being silently promoted as universally equivalent.

## Scope and data integrity

- IndicCrosslingualSTS is pinned to
  `mteb/IndicCrosslingualSTS@f0366eb5a20087355c0e131162bbed943ba54b51`: all 12 test
  configurations, 256 pairs each, macro cosine Spearman x 100, CC0-1.0.
- ArguAna is pinned to
  `mteb/arguana@c22ab2a51041ffd869aaddef7af8d8215647e41a`: 8,674 corpus items and 1,406
  queries/qrels, nDCG@10 x 100, CC-BY-SA-4.0.
- Every downloaded file was checked against its predeclared SHA-256. Task data is not
  redistributed here.
- Five pinned ArguAna relevance targets are absent from the source corpus. They are retained as
  zero-scoring queries, matching the declared contract, and the evaluator aborts if that exact
  missing-target set changes.
- The English-only MiniLM baseline has negative macro correlation on this cross-lingual STS set.
  This result is therefore a fixed numerical-preservation stress test, not a cross-lingual
  capability claim and not an official MTEB leaderboard result.

## Reproduce

```bash
uv sync --frozen
uv run armbench-minilm prepare --work-dir .armbench
uv run armbench-minilm quality-eval \
  --work-dir .armbench \
  --output-dir results/experiments/r2-bf16-task-quality \
  --batch-size 32 \
  --threads 4 \
  --code-revision "$(git rev-parse HEAD)"
```

The retained reviewer report is
[`run-31495451767/quality.md`](run-31495451767/quality.md); the machine-readable record is
[`run-31495451767/quality.json`](run-31495451767/quality.json). Verify both files with
[`SHA256SUMS`](SHA256SUMS).
