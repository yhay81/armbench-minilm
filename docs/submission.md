# Devpost submission draft

## Project name

ArmBench MiniLM

## Tagline

Reproducible FP32-to-INT8 sentence embeddings, measured and quality-checked on native Arm64.

## Inspiration

Teams evaluating CPU inference often see an optimization recommendation but not enough evidence to trust it. Model size alone does not show latency, and latency alone can hide representation drift. ArmBench MiniLM turns the whole experiment into one command and one inspectable evidence bundle.

## What it does

ArmBench MiniLM downloads an exact, public revision of the MiniLM sentence-transformer ONNX model, creates a dynamic per-channel signed INT8 version, and runs both variants with identical ONNX Runtime CPU settings. It reports model size, median and p95 latency, sentence throughput, embedding cosine preservation, and pairwise-similarity error as JSON, Markdown, and standalone HTML.

The public GitHub Actions workflow runs on a native Arm64 Ubuntu runner, so the submitted numbers are reproducible without asking reviewers to provision hardware or trust a screenshot.

## How we built it

- Python 3.12 and uv for a locked, one-command environment
- ONNX Runtime for CPU inference and dynamic quantization
- Hugging Face Hub for a revision-pinned Apache-2.0 source model
- NumPy for pooling, normalization, and fidelity metrics
- GitHub Actions `ubuntu-24.04-arm` for native Arm64 validation

Tokenization is deliberately outside the timed region. Each timed iteration includes ONNX inference, attention-mask mean pooling, and L2 normalization. Both models use four intra-op threads, one inter-op thread, the CPU execution provider, and the same authored sentence batches.

## What we optimized

The original FP32 weights used by constant-weight `MatMul` and `Gemm` operators are converted to signed INT8 with per-channel scales. Activations remain dynamically quantized at runtime. This reduces stored weights and memory traffic while preserving the same model inputs and 384-dimensional sentence-embedding interface.

## Measured result on Arm64

In the [public 100-iteration native Arm64 run](https://github.com/yhay81/armbench-minilm/actions/runs/31405378460), INT8 reduced median embedding latency by 1.80x at batch 1, 2.62x at batch 8, and 3.12x at batch 32. The geometric-mean median-latency speedup was 2.45x.

The model file shrank from 86.22 MiB to 56.04 MiB (35.0%). Across 32 authored sentences, corresponding FP32/INT8 normalized embeddings retained 0.99267173 mean cosine and 0.97647780 minimum cosine. The committed JSON contains the complete precision, checksums, machine metadata, p95 values, and workload settings.

## Fidelity guardrail

Both models embed the same 32 authored sentences. We compare corresponding normalized embeddings and every pairwise similarity. This does not claim task accuracy, but it makes numerical drift visible beside performance rather than relegating it to an assumption.

## Challenges we ran into

An optimization benchmark can accidentally measure tokenization, download time, model loading, or a different runtime configuration. We made the timing boundary explicit, pinned the model revision, stored checksums in the result, and generated all human-readable claims from the machine-readable JSON. We also report batch sizes separately because one aggregate number would conceal deployment-relevant trade-offs.

## Accomplishments that we're proud of

- One command reproduces download, quantization, measurement, fidelity checks, and reports.
- Reviewers can validate the result on a clean native Arm64 runner.
- No private data, token, pre-generated model binary, or special hardware account is required.
- The project reports limitations and quality drift alongside speedup.

## What we learned

The valuable unit of optimization is not a smaller model file; it is a reproducible deployment trade-off. Pinning provenance and measuring output fidelity are as important as choosing a quantization operator.

## What's next

Add downstream retrieval evaluation, compare thread counts and Arm instance families, and publish a small matrix of reproducible ONNX optimization recipes for other encoder architectures.

## AI assistance disclosure

OpenAI Codex materially assisted with code implementation, tests, documentation, and submission drafting. The public CI workflow executes every published measurement; AI-generated text was reviewed against those artifacts before submission.

## Links

- Source: https://github.com/yhay81/armbench-minilm
- Native Arm64 workflow: https://github.com/yhay81/armbench-minilm/actions/runs/31405378460
- Demo video: pending
