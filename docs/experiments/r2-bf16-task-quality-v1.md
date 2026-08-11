# Experiment r2-bf16-task-quality-v1

- Status: predeclared on 2026-08-11; not yet run on native Arm64
- Parent: `r2-bf16-fastmath-v1`
- Origin: agent-proposed after primary-source and license review

## Observation and hypothesis

Five independent native Arm64 runs established that enabling ONNX Runtime's
`mlas.enable_gemm_fastmath_arm64_bfloat16=1` session entry improves the FP32 model by a
median 1.3348x geometric mean. The performance gate is complete, but the authored
32-sentence cosine check is not a downstream task benchmark.

Hypothesis: the BF16 session entry preserves the task quality of each unchanged model artifact
within the predeclared semantic-similarity and retrieval tolerances. The same run will quantify
whether the existing dynamic-QInt8 artifact also remains inside those task tolerances relative to
FP32; that comparison is diagnostic and cannot retroactively change the BF16 hypothesis.

## Smallest implementation delta

Do not change either ONNX graph, tokenizer, or pooling code. Evaluate the four existing session
variants on a fixed, revision-pinned MTEB-derived gate:

1. `fp32_control`
2. `fp32_bf16_fastmath`
3. `qint8_control`
4. `qint8_bf16_fastmath`

The only difference in each same-artifact BF16 comparison remains the single ONNX Runtime session
entry. Evaluation code and reports are new; model behavior is otherwise unchanged.

## Data contract

The task data is downloaded at runtime from exact public Hugging Face Hub commits and is not
redistributed in this repository.

| Task | Pinned source | Scope | Primary score | License |
|---|---|---|---|---|
| IndicCrosslingualSTS | `mteb/IndicCrosslingualSTS@c4d2c4d658ff6dbf1d373d44cc558c9f1bb16f52` | all 12 English–Indic test configurations | macro mean Spearman correlation × 100 | CC0-1.0 |
| ArguAna | `mteb/arguana@6c1bcf74b13dfd823aff056b79d4d93e702f19c7` | complete default test corpus, queries, and qrels | nDCG@10 × 100 | CC-BY-SA-4.0 |

Every downloaded file must match its predeclared SHA-256 before evaluation. Expected cardinalities
are 12 STS configurations, 8,674 retrieval documents, 1,406 queries, and 1,406 relevance rows.
An unexpected file hash, schema, cardinality, duplicate identifier, or missing relevance target
aborts the run.

This is a pinned engineering gate, not an official MTEB leaderboard result. ArmBench fixes
`max_length=128` to match its deployment interface, whereas official leaderboard harness settings
may differ.

## Inference and metric contract

- Source model and tokenizer: `sentence-transformers/all-MiniLM-L6-v2` at
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Tokenization: fast tokenizer, dynamic batch padding, truncation enabled, `max_length=128`.
- Embedding: attention-mask mean pooling followed by row-wise L2 normalization.
- Runtime: ONNX Runtime CPUExecutionProvider, four intra-op threads, one inter-op thread, batch 32.
- STS: cosine similarity followed by average-rank Spearman correlation, reported per language pair
  and as an unweighted macro mean.
- Retrieval: cosine similarity over the complete normalized query/corpus matrices and nDCG@10.
  Score ties are broken deterministically by corpus identifier.
- Fidelity: mean cosine between corresponding candidate and control embeddings over every task
  input, reported separately for STS and retrieval.

## Predeclared decision rules

For each comparison, success requires all three conditions:

- mean corresponding-embedding cosine at least `0.99` on both tasks;
- no more than `0.5` absolute points of loss in macro STS Spearman × 100;
- no more than `1%` relative loss in ArguAna nDCG@10.

Comparisons and interpretation:

| Comparison | Purpose | Consequence if it passes |
|---|---|---|
| FP32+BF16 vs FP32 | Complete the retained FP32 BF16 promotion gate | Retain as a validated Arm64 FP32 serving option |
| QInt8+BF16 vs QInt8 | Test the small residual-GEMM optimization | Keep only with the already declared shape-specific performance restriction |
| QInt8 vs FP32 | Audit the submitted quantized artifact on downstream tasks | Add task-quality evidence to the public submission; disclose any failed task |
| QInt8+BF16 vs FP32 | Measure the combined trade-off | Inform follow-up only; no new latency headline without matched repeated timing evidence |

Reject a comparison when any threshold fails. Abort rather than score when the source contract is
violated, a metric is non-finite, or the target does not advertise Arm64 BF16 support. One native
Arm64 run is sufficient for this deterministic quality gate; the separate latency claim already
has five independent native runs.

## Reproduction target

```bash
uv run armbench-minilm quality-eval \
  --work-dir .armbench \
  --output-dir results/experiments/r2-bf16-task-quality \
  --batch-size 32 \
  --threads 4
```

The run must retain machine-readable JSON, a reviewer-readable Markdown summary, source hashes,
task cardinalities, per-variant scores, comparison deltas, and the explicit verdict.
