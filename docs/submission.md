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
- **Verify it:** [Source repository](https://github.com/yhay81/armbench-minilm) · [native Arm64 workflow run](https://github.com/yhay81/armbench-minilm/actions/runs/31405378460) · [machine-readable result](https://github.com/yhay81/armbench-minilm/blob/main/benchmarks/github-arm64-run-31405378460/benchmark.json)

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

Add downstream retrieval evaluation, compare thread counts and Arm instance families, and publish a small matrix of reproducible ONNX optimization recipes for other encoder architectures.

## Limitations

- The 32 sentences are authored workload samples, not a task-accuracy dataset.
- One GitHub-hosted virtual machine does not represent every Arm CPU or production workload.
- Dynamic quantization performance depends on model shape, batch size, thread count, runtime version, and hardware.
- The benchmark measures the embedding pipeline, not end-to-end search-service latency.

## Licensing and AI assistance disclosure

The original project code is MIT licensed. The pinned MiniLM source model is Apache-2.0. OpenAI Codex materially assisted with implementation, tests, documentation, and submission drafting; every published measurement comes from the executable public workflow.

## Links

- Submitted project: https://devpost.com/software/armbench-minilm
- Source: https://github.com/yhay81/armbench-minilm
- Native Arm64 workflow: https://github.com/yhay81/armbench-minilm/actions/runs/31405378460
- Machine-readable result: https://github.com/yhay81/armbench-minilm/blob/main/benchmarks/github-arm64-run-31405378460/benchmark.json
