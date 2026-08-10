"""Pinned inputs and authored benchmark text."""

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_FILENAME = "onnx/model.onnx"
BASELINE_NAME = "minilm-fp32.onnx"
OPTIMIZED_NAME = "minilm-qint8-arm64.onnx"
MAX_LENGTH = 128

# Authored for this benchmark. These are workload samples, not an accuracy dataset.
BENCHMARK_SENTENCES = (
    "A compact language model can make semantic search practical on CPU servers.",
    "Arm cloud instances combine energy efficiency with high core density.",
    "Dynamic quantization converts selected matrix weights from float to integers.",
    "A reproducible benchmark records the model revision and runtime configuration.",
    "Latency measures how long one request takes from start to finish.",
    "Throughput measures how many sentences the service can process each second.",
    "Output fidelity checks whether optimization preserves useful representations.",
    "Public artifacts should include licenses, setup instructions, and limitations.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "Cloud inference workloads benefit from predictable deployment automation.",
    "A pinned dependency graph makes performance experiments easier to reproduce.",
    "The benchmark compares both models on the same native Arm64 virtual machine.",
    "Model size matters when containers scale across many short-lived workers.",
    "Good optimization reports show unsuccessful trade-offs as clearly as wins.",
    "Sentence embeddings map short passages into a shared vector space.",
    "A quality guardrail prevents a faster model from hiding unacceptable drift.",
    "CI runners provide a clean machine for every recorded experiment.",
    "Median and tail latency describe different parts of the user experience.",
    "The source model is downloaded from a revision-pinned public repository.",
    "No private datasets or credentials are required to reproduce this experiment.",
    "A small command-line tool can turn optimization advice into executable evidence.",
    "The report separates measured results from assumptions and future work.",
    "CPU-only inference can be a cost-effective choice for moderate traffic.",
    "Every chart should state its unit, workload, and hardware architecture.",
    "Quantized weights reduce storage and memory bandwidth pressure.",
    "Reliable software checks shapes and input names instead of assuming an export format.",
    "Developers need one command that prepares, measures, and documents the experiment.",
    "An optimization is useful only when another engineer can validate it.",
    "Short text retrieval is a common building block in agents and search systems.",
    "ArmBench keeps tokenization outside the timed region and reports that choice.",
    "The same workload is repeated for warm-up and measured iterations.",
    "Responsible benchmarks avoid claiming universal performance from one machine.",
)
