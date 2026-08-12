# Devpost submission copy

The story below is the reviewed copy used on the public Devpost project page.

## Project name

ArmBench MiniLM

## Tagline

One command quantizes MiniLM to INT8 and proves a 2.45x latency speedup on native Arm64—with reproducible CI and embedding-fidelity guardrails.

## Project story

## ArmBench MiniLM in 30 seconds

**Track: Cloud AI**

- **Problem:** A smaller quantized model is not useful if its speedup is measured unfairly or its embeddings drift too far.
- **Solution:** ArmBench MiniLM turns FP32-to-INT8 optimization, benchmarking, fidelity checks, and reporting into one reproducible command.
- **Measured result:** On a native Arm64 runner, INT8 delivered a **2.45x geometric-mean median-latency speedup**, reduced model size by **35.0%**, and retained **0.99267173 mean FP32/INT8 embedding cosine** on the authored workload.
- **New validation:** A same-artifact FP32+BF16 option produced a repeatable **1.3348x** speedup and passed a predeclared native Arm64 STS/retrieval quality gate with embedding cosines above **0.99998**.
- **Watch it:** [74-second Full HD demo](https://youtu.be/WruMRU7M9PE)
- **Verify it:** [Source repository](https://github.com/yhay81/armbench-minilm) · [submitted native Arm64 run](https://github.com/yhay81/armbench-minilm/actions/runs/31405378460) · [BF16 task-quality run](https://github.com/yhay81/armbench-minilm/actions/runs/31495451767) · [machine-readable result](https://github.com/yhay81/armbench-minilm/blob/main/benchmarks/github-arm64-run-31405378460/benchmark.json)

## Inspiration

Teams evaluating CPU inference are often told to quantize a model, but a smaller file alone is not enough evidence to trust a deployment. Latency tests can accidentally include tokenization or model loading, and a headline speedup can hide unacceptable representation drift.

ArmBench MiniLM was created to make the whole trade-off visible: speed, size, fidelity, provenance, and limitations in one inspectable evidence bundle.

## What it does

ArmBench MiniLM:

1. downloads an exact public revision of `sentence-transformers/all-MiniLM-L6-v2`;
2. converts constant-weight `MatMul` and `Gemm` operations from FP32 to dynamic per-channel signed INT8;
3. runs both models with identical ONNX Runtime CPU settings;
4. measures model size, median and p95 latency, and sentence throughput;
5. checks corresponding embedding cosine and every pairwise-similarity error; and
6. emits JSON, Markdown, and standalone HTML reports.

The public GitHub Actions workflow runs on a native Arm64 Ubuntu runner, so the submitted numbers are reproducible without asking reviewers to provision hardware or trust a screenshot.

## How we built it

- Python and uv for a locked, one-command environment
- ONNX Runtime for CPU inference and dynamic quantization
- Hugging Face Hub for a revision-pinned Apache-2.0 source model
- NumPy for pooling, normalization, and fidelity metrics
- GitHub Actions `ubuntu-24.04-arm` for native Arm64 validation

Tokenization is deliberately outside the timed region. Each timed iteration includes ONNX inference, attention-mask mean pooling, and L2 normalization. Both models use four intra-op threads, one inter-op thread, the CPU execution provider, and the same authored sentence batches.

## What we optimized

The original FP32 weights used by constant-weight `MatMul` and `Gemm` operators are converted to signed INT8 with per-channel scales. Activations remain dynamically quantized at runtime. This reduces stored weights and memory traffic while preserving the same model inputs and 384-dimensional sentence-embedding interface.

## Measured results on native Arm64

The public 100-iteration run used four Arm64 CPU cores, ONNX Runtime 1.28.0, ten warm-ups, and 100 measured iterations per model and batch.

| Batch | FP32 median | INT8 median | Speedup | INT8 throughput |
|---:|---:|---:|---:|---:|
| 1 | 2.928 ms | 1.629 ms | **1.80x** | 613.7 sentences/s |
| 8 | 19.331 ms | 7.376 ms | **2.62x** | 1,084.6 sentences/s |
| 32 | 74.264 ms | 23.815 ms | **3.12x** | 1,343.7 sentences/s |

Across all three batches, INT8 produced a **2.45x geometric-mean median-latency speedup**. The model file fell from 86.22 MiB to 56.04 MiB, a **35.0% reduction**.

The optimized model retained **0.99267173 mean corresponding-embedding cosine** (minimum 0.97647780) across 32 authored workload sentences. Mean pairwise-similarity absolute error was 0.01012334.

## New downstream validation checkpoint

After freezing the original submission result, we tested four unchanged artifacts/sessions on a
revision-pinned, hash-checked MTEB-derived gate: FP32, FP32+BF16, QInt8, and QInt8+BF16. The gate
uses all 12 IndicCrosslingualSTS test configurations and the full 8,674-document, 1,406-query
ArguAna test set at the project's 128-token deployment limit.

FP32+BF16 passed every predeclared condition. Relative to FP32, STS changed from -6.2773 to
-6.2677, ArguAna nDCG@10 changed from 48.9917 to 49.0337, and corresponding-embedding cosine was
0.99998246 on STS and 0.99998768 on retrieval. Together with five independent native performance
runs (1.3348x median geometric-mean speedup, 0.38% run CV), this makes FP32+BF16 a validated Arm64
FP32 serving option.

The submitted QInt8 artifact did not lose either task score, but its 0.98911148 STS
corresponding-embedding cosine narrowly missed the deliberately strict 0.99 gate. We retain the
immutable 2.45x speed result and disclose this boundary instead of claiming universal numerical
equivalence. The English-only source model scores negatively on the English–Indic STS set, so
that part is a preservation stress test—not a cross-lingual capability claim or an official MTEB
leaderboard result. [Read the retained quality evidence.](https://github.com/yhay81/armbench-minilm/tree/main/benchmarks/r2-bf16-task-quality-af45559)

## Fidelity guardrail

Both models embed the same 32 authored sentences. We compare corresponding normalized embeddings and every pairwise similarity. This is a numerical drift check, not a task-accuracy claim. It puts fidelity beside performance instead of leaving it as an assumption.

## Reproduce it

No private dataset, Hugging Face token, or special hardware account is required.

```bash
git clone https://github.com/yhay81/armbench-minilm.git
cd armbench-minilm
uv sync --frozen
uv run armbench-minilm all --work-dir .armbench --output-dir results
```

The command downloads the revision-pinned FP32 ONNX model, creates the INT8 derivative locally, benchmarks both, evaluates drift, and writes `benchmark.json`, `report.md`, and `report.html`. To reproduce the submitted machine, fork the repository and run **Native Arm64 benchmark** from GitHub Actions.

## Challenges we ran into

An optimization benchmark can accidentally measure tokenization, download time, model loading, or a different runtime configuration. We made the timing boundary explicit, pinned the model revision, stored checksums in the result, and generated all human-readable claims from the machine-readable JSON. We also report batch sizes separately because one aggregate number would conceal deployment-relevant trade-offs.

## What makes it useful to developers

- One command reproduces download, quantization, measurement, fidelity checks, and reports.
- Reviewers can validate the result on a clean native Arm64 runner.
- No private data, token, pre-generated model binary, or special hardware account is required.
- The project reports limitations and quality drift alongside speedup.

## What we learned

The valuable unit of optimization is not a smaller model file; it is a reproducible deployment trade-off. Pinning provenance and measuring output fidelity are as important as choosing a quantization operator.

## What's next

Fuse transformer, pooling, and normalization graph operations; compare thread counts and Arm
instance families; then evaluate a calibrated static S8S8 candidate against the same pinned task
gate.

## Limitations

- The 32 authored sentences are a fast numerical drift check; downstream evidence comes from a separate pinned engineering gate, not an official MTEB run.
- The English-only source model's cross-lingual STS score is negative, so the STS slice supports preservation testing only, not a cross-lingual capability claim.
- Five pinned ArguAna qrels point to documents absent from the source corpus and score zero under the declared contract.
- One GitHub-hosted virtual machine does not represent every Arm CPU or production workload.
- Dynamic quantization performance depends on model shape, batch size, thread count, runtime version, and hardware.
- The benchmark measures the embedding pipeline, not end-to-end search-service latency.

## Licensing and AI assistance disclosure

The original project code is MIT licensed. The pinned MiniLM source model is Apache-2.0. OpenAI Codex materially assisted with implementation, tests, documentation, and submission drafting; every published measurement comes from the executable public workflow.

## Links

- Submitted project: https://devpost.com/software/armbench-minilm
- Source: https://github.com/yhay81/armbench-minilm
- Native Arm64 workflow: https://github.com/yhay81/armbench-minilm/actions/runs/31405378460
- BF16 task-quality workflow: https://github.com/yhay81/armbench-minilm/actions/runs/31495451767
- Machine-readable result: https://github.com/yhay81/armbench-minilm/blob/main/benchmarks/github-arm64-run-31405378460/benchmark.json
