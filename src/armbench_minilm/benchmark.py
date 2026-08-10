"""Native CPU benchmark orchestration for the baseline and quantized models."""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import cycle, islice
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import onnxruntime as ort
import psutil
from numpy.typing import NDArray
from transformers import AutoTokenizer

from armbench_minilm.constants import (
    BENCHMARK_SENTENCES,
    MAX_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
)
from armbench_minilm.metrics import mean_pool, normalize_rows, quality_metrics, summarize_latencies
from armbench_minilm.models import ModelPaths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_session(model_path: Path, *, threads: int) -> tuple[ort.InferenceSession, float]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    started = perf_counter()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return session, (perf_counter() - started) * 1000.0


def _batch(sentences: Sequence[str], batch_size: int) -> list[str]:
    return list(islice(cycle(sentences), batch_size))


def _feeds(
    tokenizer: Any,
    session: ort.InferenceSession,
    sentences: Sequence[str],
) -> dict[str, NDArray[Any]]:
    encoded = tokenizer(
        list(sentences),
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="np",
    )
    expected = {item.name for item in session.get_inputs()}
    feeds = {name: np.asarray(value) for name, value in encoded.items() if name in expected}
    missing = expected.difference(feeds)
    if missing:
        raise ValueError(f"tokenizer did not produce required ONNX inputs: {sorted(missing)}")
    return feeds


def _embed(
    session: ort.InferenceSession,
    feeds: dict[str, NDArray[Any]],
) -> NDArray[np.float32]:
    outputs = session.run(None, feeds)
    embeddings = np.asarray(outputs[0])
    if embeddings.ndim == 3:
        if "attention_mask" not in feeds:
            raise ValueError("token-level output requires an attention_mask for mean pooling")
        embeddings = mean_pool(embeddings, np.asarray(feeds["attention_mask"]))
    elif embeddings.ndim != 2:
        raise ValueError(f"unsupported first ONNX output shape: {embeddings.shape}")
    return normalize_rows(embeddings)


def _measure(
    session: ort.InferenceSession,
    feeds: dict[str, NDArray[Any]],
    *,
    warmups: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmups):
        _embed(session, feeds)

    latencies_ms: list[float] = []
    for _ in range(iterations):
        started = perf_counter()
        _embed(session, feeds)
        latencies_ms.append((perf_counter() - started) * 1000.0)
    return summarize_latencies(latencies_ms, batch_size=int(feeds["input_ids"].shape[0]))


def _model_metadata(path: Path, *, load_ms: float) -> dict[str, Any]:
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "session_load_ms": load_ms,
    }


def _machine_metadata() -> dict[str, Any]:
    return {
        "architecture": platform.machine(),
        "processor": platform.processor() or "not reported",
        "operating_system": platform.platform(),
        "python": sys.version.split()[0],
        "onnxruntime": ort.__version__,
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "memory_bytes": psutil.virtual_memory().total,
        "github_runner_name": os.getenv("RUNNER_NAME"),
        "github_runner_os": os.getenv("RUNNER_OS"),
        "github_runner_arch": os.getenv("RUNNER_ARCH"),
        "github_run_id": os.getenv("GITHUB_RUN_ID"),
        "github_sha": os.getenv("GITHUB_SHA"),
        "execution_provider": "CPUExecutionProvider",
    }


def run_benchmark(
    paths: ModelPaths,
    *,
    batch_sizes: Sequence[int],
    warmups: int,
    iterations: int,
    threads: int,
) -> dict[str, Any]:
    """Benchmark both models on one machine and return a serializable result."""

    if not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    if warmups < 0 or iterations < 1 or threads < 1:
        raise ValueError("warmups must be non-negative; iterations and threads must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=True,
    )
    baseline_session, baseline_load_ms = _create_session(paths.baseline, threads=threads)
    optimized_session, optimized_load_ms = _create_session(paths.optimized, threads=threads)

    quality_feeds = _feeds(tokenizer, baseline_session, BENCHMARK_SENTENCES)
    baseline_embeddings = _embed(baseline_session, quality_feeds)
    optimized_embeddings = _embed(optimized_session, quality_feeds)

    batches: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        sentences = _batch(BENCHMARK_SENTENCES, batch_size)
        baseline_feeds = _feeds(tokenizer, baseline_session, sentences)
        optimized_feeds = _feeds(tokenizer, optimized_session, sentences)
        baseline_metrics = _measure(
            baseline_session,
            baseline_feeds,
            warmups=warmups,
            iterations=iterations,
        )
        optimized_metrics = _measure(
            optimized_session,
            optimized_feeds,
            warmups=warmups,
            iterations=iterations,
        )
        speedup = baseline_metrics["median_ms"] / optimized_metrics["median_ms"]
        batches.append(
            {
                "batch_size": batch_size,
                "baseline": baseline_metrics,
                "optimized": optimized_metrics,
                "median_latency_speedup": speedup,
                "throughput_gain_percent": (
                    optimized_metrics["sentences_per_second"]
                    / baseline_metrics["sentences_per_second"]
                    - 1.0
                )
                * 100.0,
            }
        )

    baseline_size = paths.baseline.stat().st_size
    optimized_size = paths.optimized.stat().st_size
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": "Apache-2.0",
        },
        "optimization": {
            "method": "ONNX Runtime dynamic per-channel signed INT8 weight quantization",
            "operator_types": ["MatMul", "Gemm"],
            "timing_boundary": (
                "ONNX inference plus mean pooling and L2 normalization; tokenization excluded"
            ),
        },
        "configuration": {
            "batch_sizes": list(batch_sizes),
            "warmups_per_model_and_batch": warmups,
            "measured_iterations_per_model_and_batch": iterations,
            "intra_op_threads": threads,
            "inter_op_threads": 1,
            "max_token_length": MAX_LENGTH,
            "quality_sentence_count": len(BENCHMARK_SENTENCES),
        },
        "machine": _machine_metadata(),
        "models": {
            "baseline": _model_metadata(paths.baseline, load_ms=baseline_load_ms),
            "optimized": _model_metadata(paths.optimized, load_ms=optimized_load_ms),
        },
        "quality": quality_metrics(baseline_embeddings, optimized_embeddings),
        "batches": batches,
        "summary": {
            "model_size_reduction_percent": (1.0 - optimized_size / baseline_size) * 100.0,
            "geometric_mean_latency_speedup": float(
                np.exp(np.mean(np.log([item["median_latency_speedup"] for item in batches])))
            ),
        },
    }
