"""Native CPU benchmark orchestration for the baseline and quantized models."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
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
from armbench_minilm.metrics import (
    bootstrap_speedup_ci,
    mean_pool,
    normalize_rows,
    quality_metrics,
    summarize_latencies,
)
from armbench_minilm.models import ModelPaths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_session(
    model_path: Path,
    *,
    threads: int,
    profile_prefix: Path | None = None,
) -> tuple[ort.InferenceSession, float]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile_prefix is not None:
        profile_prefix.parent.mkdir(parents=True, exist_ok=True)
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix.resolve())
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
    *,
    fixed_sequence_length: int | None = None,
) -> dict[str, NDArray[Any]]:
    max_length = fixed_sequence_length or MAX_LENGTH
    encoded = tokenizer(
        list(sentences),
        padding="max_length" if fixed_sequence_length is not None else True,
        truncation=True,
        max_length=max_length,
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


def _measure_samples(
    session: ort.InferenceSession,
    feeds: dict[str, NDArray[Any]],
    iterations: int,
) -> list[float]:
    latencies_ms: list[float] = []
    for _ in range(iterations):
        started = perf_counter()
        _embed(session, feeds)
        latencies_ms.append((perf_counter() - started) * 1000.0)
    return latencies_ms


def _trial_sizes(iterations: int, measurement_blocks: int) -> list[int]:
    if measurement_blocks < 1:
        raise ValueError("measurement_blocks must be positive")
    if iterations < measurement_blocks:
        raise ValueError("iterations must be at least measurement_blocks")
    base, remainder = divmod(iterations, measurement_blocks)
    return [base + (1 if index < remainder else 0) for index in range(measurement_blocks)]


def _measure_randomized_blocks(
    sessions: Mapping[str, ort.InferenceSession],
    feeds_by_model: Mapping[str, dict[str, NDArray[Any]]],
    *,
    warmups: int,
    iterations: int,
    measurement_blocks: int,
    seed: int,
) -> tuple[dict[str, list[float]], list[dict[str, Any]]]:
    """Warm both models and measure randomized A/B blocks with raw samples."""

    names = list(sessions)
    if set(names) != set(feeds_by_model):
        raise ValueError("sessions and feeds_by_model must have the same model names")
    rng = random.Random(seed)
    for _ in range(warmups):
        order = names.copy()
        rng.shuffle(order)
        for name in order:
            _embed(sessions[name], feeds_by_model[name])

    collected = {name: [] for name in names}
    blocks: list[dict[str, Any]] = []
    for block_index, count in enumerate(_trial_sizes(iterations, measurement_blocks)):
        order = names.copy()
        rng.shuffle(order)
        samples_by_model: dict[str, list[float]] = {}
        for name in order:
            samples = _measure_samples(sessions[name], feeds_by_model[name], count)
            samples_by_model[name] = samples
            collected[name].extend(samples)
        blocks.append(
            {
                "block_index": block_index,
                "order": order,
                "iterations_per_model": count,
                "samples_ms": samples_by_model,
            }
        )
    return collected, blocks


def _model_metadata(path: Path, *, load_ms: float) -> dict[str, Any]:
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "session_load_ms": load_ms,
    }


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _linux_cpu_metadata() -> dict[str, Any]:
    cpuinfo = _read_text(Path("/proc/cpuinfo"))
    fields: dict[str, str] = {}
    if cpuinfo:
        wanted = {
            "model name",
            "hardware",
            "cpu implementer",
            "cpu architecture",
            "cpu variant",
            "cpu part",
            "cpu revision",
            "features",
        }
        for line in cpuinfo.splitlines():
            if not line.strip() and fields:
                break
            key, separator, value = line.partition(":")
            normalized = key.strip().lower()
            if separator and normalized in wanted:
                fields[normalized.replace(" ", "_")] = value.strip()

    features = fields.pop("features", "")
    cache_topology: list[dict[str, str]] = []
    cache_root = Path("/sys/devices/system/cpu/cpu0/cache")
    if cache_root.is_dir():
        for index_path in sorted(cache_root.glob("index*")):
            entry = {
                key: value
                for key, filename in (
                    ("level", "level"),
                    ("type", "type"),
                    ("size", "size"),
                    ("shared_cpu_list", "shared_cpu_list"),
                )
                if (value := _read_text(index_path / filename)) is not None
            }
            if entry:
                cache_topology.append(entry)

    return {
        "identity": fields,
        "features": sorted(features.split()),
        "cache_topology": cache_topology,
        "scaling_governor": _read_text(
            Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
        ),
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
        "github_runner_image_os": os.getenv("ImageOS"),
        "github_runner_image_version": os.getenv("ImageVersion"),
        "execution_provider": "CPUExecutionProvider",
        "available_execution_providers": ort.get_available_providers(),
        "onnxruntime_build_info": ort.get_build_info(),
        "linux_cpu": _linux_cpu_metadata() if sys.platform.startswith("linux") else None,
    }


def _summarize_profile(profile_path: Path) -> dict[str, Any]:
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    by_operator: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"duration_us": 0.0, "calls": 0}
    )
    for event in events:
        args = event.get("args", {})
        operator = args.get("op_name")
        duration = event.get("dur")
        if event.get("cat") != "Node" or not operator or not isinstance(duration, int | float):
            continue
        by_operator[operator]["duration_us"] += float(duration)
        by_operator[operator]["calls"] += 1

    operators = [
        {
            "operator": operator,
            "duration_us": values["duration_us"],
            "calls": values["calls"],
        }
        for operator, values in by_operator.items()
    ]
    operators.sort(key=lambda item: float(item["duration_us"]), reverse=True)
    return {
        "profile_file": profile_path.name,
        "node_duration_us": sum(float(item["duration_us"]) for item in operators),
        "operators": operators,
    }


def _profile_model_case(
    model_path: Path,
    tokenizer: Any,
    sentences: Sequence[str],
    *,
    model_name: str,
    batch_size: int,
    sequence_length: int,
    threads: int,
    profile_dir: Path,
) -> dict[str, Any]:
    prefix = profile_dir / f"{model_name}-b{batch_size}-s{sequence_length}"
    session, _ = _create_session(model_path, threads=threads, profile_prefix=prefix)
    feeds = _feeds(
        tokenizer,
        session,
        sentences,
        fixed_sequence_length=sequence_length,
    )
    _embed(session, feeds)
    for _ in range(3):
        _embed(session, feeds)
    profile_path = Path(session.end_profiling())
    return _summarize_profile(profile_path)


def run_benchmark(
    paths: ModelPaths,
    *,
    batch_sizes: Sequence[int],
    warmups: int,
    iterations: int,
    threads: int,
    sequence_lengths: Sequence[int] = (128,),
    measurement_blocks: int = 5,
    random_seed: int = 20260811,
    bootstrap_resamples: int = 2_000,
    profile_dir: Path | None = None,
) -> dict[str, Any]:
    """Benchmark both models on one machine and return a serializable result."""

    if not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    if not sequence_lengths or any(
        length < 1 or length > MAX_LENGTH for length in sequence_lengths
    ):
        raise ValueError(f"sequence_lengths must be between 1 and {MAX_LENGTH}")
    if warmups < 0 or iterations < 1 or threads < 1:
        raise ValueError("warmups must be non-negative; iterations and threads must be positive")
    _trial_sizes(iterations, measurement_blocks)
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")

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
    sessions = {"baseline": baseline_session, "optimized": optimized_session}
    case_index = 0
    for batch_size in batch_sizes:
        for sequence_length in sequence_lengths:
            sentences = _batch(BENCHMARK_SENTENCES, batch_size)
            baseline_feeds = _feeds(
                tokenizer,
                baseline_session,
                sentences,
                fixed_sequence_length=sequence_length,
            )
            optimized_feeds = _feeds(
                tokenizer,
                optimized_session,
                sentences,
                fixed_sequence_length=sequence_length,
            )
            case_seed = random_seed + case_index * 1_009
            samples, blocks = _measure_randomized_blocks(
                sessions,
                {"baseline": baseline_feeds, "optimized": optimized_feeds},
                warmups=warmups,
                iterations=iterations,
                measurement_blocks=measurement_blocks,
                seed=case_seed,
            )
            baseline_metrics = summarize_latencies(
                samples["baseline"],
                batch_size=batch_size,
                bootstrap_seed=case_seed + 1,
                bootstrap_resamples=bootstrap_resamples,
            )
            optimized_metrics = summarize_latencies(
                samples["optimized"],
                batch_size=batch_size,
                bootstrap_seed=case_seed + 2,
                bootstrap_resamples=bootstrap_resamples,
            )
            speedup = baseline_metrics["median_ms"] / optimized_metrics["median_ms"]
            speedup_confidence = bootstrap_speedup_ci(
                samples["baseline"],
                samples["optimized"],
                seed=case_seed + 3,
                resamples=bootstrap_resamples,
            )
            batches.append(
                {
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "random_seed": case_seed,
                    "baseline": baseline_metrics,
                    "optimized": optimized_metrics,
                    "median_latency_speedup": speedup,
                    **speedup_confidence,
                    "throughput_gain_percent": (
                        optimized_metrics["sentences_per_second"]
                        / baseline_metrics["sentences_per_second"]
                        - 1.0
                    )
                    * 100.0,
                    "measurement_blocks": blocks,
                }
            )
            case_index += 1

    profiles: list[dict[str, Any]] = []
    if profile_dir is not None:
        profile_dir = profile_dir.resolve()
        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                sentences = _batch(BENCHMARK_SENTENCES, batch_size)
                profiles.append(
                    {
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "baseline": _profile_model_case(
                            paths.baseline,
                            tokenizer,
                            sentences,
                            model_name="baseline",
                            batch_size=batch_size,
                            sequence_length=sequence_length,
                            threads=threads,
                            profile_dir=profile_dir,
                        ),
                        "optimized": _profile_model_case(
                            paths.optimized,
                            tokenizer,
                            sentences,
                            model_name="optimized",
                            batch_size=batch_size,
                            sequence_length=sequence_length,
                            threads=threads,
                            profile_dir=profile_dir,
                        ),
                    }
                )

    baseline_size = paths.baseline.stat().st_size
    optimized_size = paths.optimized.stat().st_size
    return {
        "schema_version": 2,
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
            "sequence_lengths": list(sequence_lengths),
            "warmups_per_model_and_batch": warmups,
            "measured_iterations_per_model_and_batch": iterations,
            "measurement_blocks_per_case": measurement_blocks,
            "random_seed": random_seed,
            "bootstrap_resamples": bootstrap_resamples,
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
        "profiles": profiles,
        "summary": {
            "model_size_reduction_percent": (1.0 - optimized_size / baseline_size) * 100.0,
            "geometric_mean_latency_speedup": float(
                np.exp(np.mean(np.log([item["median_latency_speedup"] for item in batches])))
            ),
            "maximum_median_ci95_half_width_percent": max(
                max(
                    item["baseline"]["median_ci95_half_width_percent"],
                    item["optimized"]["median_ci95_half_width_percent"],
                )
                for item in batches
            ),
            "tail_spike_cases": [
                {
                    "batch_size": item["batch_size"],
                    "sequence_length": item["sequence_length"],
                    "model": model_name,
                    "p95_to_median_ratio": item[model_name]["p95_ms"]
                    / item[model_name]["median_ms"],
                }
                for item in batches
                for model_name in ("baseline", "optimized")
                if item[model_name]["p95_ms"] / item[model_name]["median_ms"] > 1.5
            ],
        },
    }
