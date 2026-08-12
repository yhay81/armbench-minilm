# Agent execution brief — ArmBench R2 graph-overhead sprint

Use this document as the complete task prompt for the next implementation agent.

## Objective

Produce one reviewable, reproducible R2 checkpoint that reduces ArmBench MiniLM's
non-kernel inference overhead on native Arm64 without changing the immutable submitted
2.45x INT8 headline. The primary candidate must combine:

1. offline ONNX Runtime transformer optimization;
2. attention-mask mean pooling and L2 normalization inside the ONNX graph; and
3. a controlled comparison of ordinary `session.run` with reusable/preallocated I/O.

Do not attempt the entire R2–R7 roadmap. Finish and validate this checkpoint before
starting static activation quantization, custom runtime builds, INT4, or architecture changes.

## Required context

Work from `D:\ghq\github.com\yhay81\kaggle` and read, in order:

1. the repository `AGENTS.md`;
2. `docs/competition-agent-workflow.md`;
3. `projects/armbench-minilm/docs/performance-roadmap.md`;
4. `projects/armbench-minilm/docs/experiments/r2-bf16-fastmath-v1.md`;
5. `projects/armbench-minilm/docs/experiments/r2-bf16-task-quality-v1.md`;
6. the current `benchmark.py`, `models.py`, `quality.py`, CLI, tests, and Arm64 workflow; and
7. the retained R0, R1, and R2 evidence linked from the project README.

Before editing, inspect the shared worktree and current branch. Preserve concurrent user/agent
changes, work directly on `main`, stage only `projects/armbench-minilm`, and follow the repository's
checkpoint commit/push cadence. Never force-push or automate a Devpost submission.

## Immutable baseline

- Submitted native run: `benchmarks/github-arm64-run-31405378460/benchmark.json`
- Submitted headline: 2.45x geometric-mean median-latency speedup for dynamic QInt8
- Model-size reduction: 35.0%
- Authored-workload mean FP32/QInt8 embedding cosine: 0.99267173
- Validated FP32+BF16 checkpoint: 1.3348x five-run median geometric-mean speedup,
  0.38% run CV, with the pinned STS/retrieval quality gate passing
- Benchmark contract: fixed batch/sequence shapes, balanced randomized A/B blocks,
  per-block re-warm, raw wall/process-CPU samples, deterministic bootstrap intervals,
  separate profiles, and immutable evidence by commit and workflow run

New results must be reported as a separate experimental checkpoint. Do not silently replace,
rewrite, or reinterpret any retained evidence.

## Why this is the next best experiment

The current `_embed` path calls ONNX Runtime, copies token embeddings to NumPy, performs
attention-mask mean pooling, then performs row-wise L2 normalization. The current-scope model file
is already only about 0.6% above its QInt8 initializer-payload bound, so repeating weight-only
quantization cannot materially improve size. The next credible latency gain is therefore the
measured fixed overhead and data movement around the optimized kernels.

ONNX Runtime's official transformer optimizer provides offline transformer-specific fusions that
may not be applied at session load, especially when dynamic axes block shape-dependent rewrites.
Its official tuning guidance also exposes threading and I/O controls. These should be exhausted
before adding a riskier precision change.

## Phase 0 — preregister the experiment

Before implementing the candidate, add
`docs/experiments/r2-graph-fusion-v1.md` containing:

- exact baseline and candidate artifact definitions;
- graph-construction order;
- fixed shape grid and thread settings;
- timing boundaries for encoder, output copy, pooling, normalization, and end-to-end embedding;
- numerical and downstream quality thresholds;
- performance promotion gates;
- failure/rollback rules; and
- the exact source revisions, runtime versions, and workflow entry point.

Commit and push this contract before viewing native performance results. Do not weaken thresholds
after observing a candidate.

## Phase 1 — characterize the graph before changing it

Create a deterministic graph-inspection artifact for the pinned FP32 and current QInt8 models:

- opset and symbolic/dynamic dimensions;
- operator and initializer counts;
- attention, GELU, layer-normalization, and skip-layer-normalization fusion counts;
- output names, dtypes, and shapes;
- optimized graph SHA-256; and
- per-node profile time for the existing fixed grid.

Run ONNX Runtime symbolic shape inference and the transformer optimizer with parameters derived
from the pinned model configuration. Do not assume the MiniLM hidden size or attention-head count;
read and record them from the pinned revision. Save the optimized graph separately and compare its
node/fusion inventory with both the source graph and the graph that ORT serializes after
`ORT_ENABLE_ALL` session optimization.

If the offline optimizer makes no material graph change, retain that negative result and continue
with fused embedding post-processing; do not manufacture a fusion claim.

## Phase 2 — fuse the embedding contract into ONNX

Add a deterministic graph builder, preferably in a new focused module such as
`src/armbench_minilm/graph_opt.py`. Append an output path equivalent to the current Python logic:

1. cast/reshape the attention mask safely for broadcasting;
2. mask token embeddings;
3. reduce the token axis and divide by a clipped nonzero mask sum;
4. compute the row L2 norm with a documented epsilon; and
5. divide to return normalized `[batch, 384]` embeddings.

Requirements:

- Keep the existing FP32 and submitted QInt8 artifacts immutable.
- Produce separately named FP32-fused and QInt8-fused candidates.
- Preserve dynamic-batch portability; fixed-shape candidates may be additional artifacts, not the
  only output.
- Do not time mismatched semantics. Baseline and candidate comparisons must both return the same
  normalized sentence-embedding contract.
- Update the runner so a graph-normalized output is not normalized again in Python.
- Record every graph/input/model/config SHA-256 in result JSON.
- Make generation idempotent and fail clearly on unexpected graph outputs or dimensions.

Add unit tests for graph structure, repeatable hashes where feasible, zero/fully padded mask
handling, variable batch/sequence shapes, and numerical equivalence against the existing
`mean_pool` plus `normalize_rows` path.

## Phase 3 — separate and reduce runtime overhead

Instrument without contaminating the promoted latency samples. Report at least:

- `session.run` including output allocation/copy;
- Python pooling only;
- Python normalization only;
- fused-graph end-to-end embedding;
- session creation and first-inference cold start; and
- reusable I/O binding/preallocation where the Python API and fixed output shape make it safe.

Compare a minimal native harness or `onnxruntime_perf_test` with the Python harness to identify
language/runtime overhead, but do not combine numbers from different timing boundaries.

After the primary four-thread result is stable, sweep threads `1, 2, 4`, affinity where supported,
spinning, spin duration/backoff, and the dynamic thread-pool cost model. Change one factor at a
time and retain raw samples. Never tune on one shape and imply that it wins universally.

## Phase 4 — native Arm64 validation

Add a dedicated workflow-dispatch experiment job rather than modifying the historical result.
For every promoted candidate:

- use the same clean `ubuntu-24.04-arm` runner and pinned environment;
- run the complete `(batch 1, 8, 32) × (sequence 16, 32, 64, 128)` grid;
- use five balanced randomized measurement blocks and at least 100 measured iterations per shape;
- preserve all raw wall/process-CPU samples and profiles;
- run at least five independent workflow executions before a promoted claim;
- aggregate runs by immutable commit and run ID; and
- run the existing pinned STS/retrieval quality evaluation on the exact candidate artifact.

Quality gates:

- graph-fusion numerical check: mean corresponding-embedding cosine at least `0.99999` against the
  unfused artifact, with maximum absolute error and minimum cosine reported;
- project gate: no more than 0.5 absolute points of STS loss;
- project gate: no more than 1% relative retrieval-score loss;
- identical tokenizer, truncation, mask, pooling, normalization, and output dimension; and
- no unexplained NaN, infinity, fully padded row, or fallback behavior.

Performance promotion gates:

- five-run geometric-mean latency speedup greater than `1.03x` on the predeclared target grid;
- 95% speedup confidence interval lower bound above `1.00x`;
- run-to-run coefficient of variation below 3%; and
- no unexplained p95 spike above 1.5x median.

If the candidate fails a gate, retain and label the negative result. Do not update the README
headline or Devpost copy.

## Secondary work packages — only after R2 passes

### A. Static S8S8 Pareto candidate

Compare dynamic QInt8 with one predeclared static S8S8 QDQ candidate. Optimize the transformer
graph before quantization and use ONNX Runtime quantization debugging to identify sensitive nodes.
The calibration corpus must be pinned, licensed, representative, and disjoint from every quality
test split. Never calibrate on the retained STS or ArguAna evaluation test examples. Measure load
time, size, latency, throughput, memory, and the same task gate.

### B. Arm runtime/kernel attribution

A/B test the current runtime with `mlas.disable_kleidiai` only after verifying that the key exists
in the pinned ONNX Runtime source/version. Record CPU features such as DotProd, I8MM, SVE/SVE2, or
SME/SME2 and do not combine different CPUs into one headline. A custom ACL Execution Provider
build is secondary because the official ACL EP remains community-maintained/preview; it is useful
only if compute-heavy nodes are actually assigned to ACL and build provenance is complete.

### C. XNNPACK feasibility

Treat XNNPACK as a capability experiment, not a promised optimization. Official support currently
limits Gemm and MatMul to 2D. Capture provider assignment and stop if the compute-heavy transformer
nodes fall back to CPU EP. If tested, follow the official recommendation to avoid contention:
ORT intra-op threads 1, ORT spinning disabled, and a separately sized XNNPACK thread pool.

## Required deliverables

- preregistered experiment contract;
- deterministic graph optimizer/fuser and CLI entry point;
- focused unit tests;
- machine-readable graph inventory and result schema;
- dedicated native Arm64 experiment workflow;
- retained local and native evidence with SHA-256 and run IDs;
- a concise experiment README stating pass/fail and limitations;
- updates to `performance-roadmap.md` only after evidence exists; and
- a final handoff listing changed files, commands, checks, commits, workflow runs, and the next
  unresolved bottleneck.

Run the relevant project checks, then the repository gates required by `AGENTS.md`. At minimum:

```powershell
uv run ruff check .
uv run ty check
uv run pytest -q
uv run ai-hub-workbench validate projects/armbench-minilm
```

## Stop conditions

Stop and document instead of improvising when:

- source/model/config provenance cannot be pinned;
- graph semantics or output dimensions are ambiguous;
- calibration would overlap evaluation test data;
- native runs are unstable beyond the preregistered limits;
- a candidate needs a paid runner or external service not already authorized;
- a quality gate fails; or
- a public claim would require replacing immutable evidence or automating a competition submission.

## Primary technical references

- [ONNX Runtime transformer optimizer](https://onnxruntime.ai/docs/performance/transformers-optimization.html)
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ONNX Runtime performance tuning](https://onnxruntime.ai/docs/performance/tune-performance/)
- [ONNX Runtime thread management](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
- [ONNX Runtime XNNPACK Execution Provider](https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html)
- [ONNX Runtime ACL Execution Provider](https://onnxruntime.ai/docs/execution-providers/community-maintained/ACL-ExecutionProvider.html)
- [Arm KleidiAI](https://github.com/ARM-software/kleidiai)
