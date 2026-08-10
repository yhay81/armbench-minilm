"""Exact-shape ONNX Runtime MatMul benchmarks for the MiniLM target scope."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, process_time
from typing import Any

import numpy as np
import onnx
from numpy.typing import NDArray
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from armbench_minilm.benchmark import (
    INTRA_OP_SPIN_BACKOFF_MAX,
    INTRA_OP_SPIN_DURATION_US,
    _create_session,
    _machine_metadata,
    _sha256,
    _trial_sizes,
)
from armbench_minilm.metrics import bootstrap_median_ci


def _target_shape_counts(model_path: Path) -> Counter[tuple[int, int]]:
    model = onnx.load(model_path.resolve(), load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    counts: Counter[tuple[int, int]] = Counter()
    for node in model.graph.node:
        if (
            node.op_type not in {"MatMul", "Gemm"}
            or len(node.input) < 2
            or node.input[1] not in initializers
        ):
            continue
        dimensions = tuple(int(value) for value in initializers[node.input[1]].dims)
        if len(dimensions) != 2:
            raise ValueError(f"target weight is not a matrix: {node.name}: {dimensions}")
        if node.op_type != "MatMul":
            raise ValueError(
                "exact-shape kernel benchmark currently requires MatMul targets; "
                f"found {node.op_type}: {node.name}"
            )
        counts[(dimensions[0], dimensions[1])] += 1
    if not counts:
        raise ValueError("baseline graph has no constant-weight MatMul targets")
    return counts


def _write_matmul_pair(
    directory: Path,
    *,
    k: int,
    n: int,
    seed: int,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    weight = rng.uniform(-0.05, 0.05, size=(k, n)).astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"], name="target/MatMul")],
        f"exact-matmul-{k}x{n}",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, k])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, n])],
        [numpy_helper.from_array(weight, name="weight")],
    )
    model = helper.make_model(
        graph,
        producer_name="armbench-minilm",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 10
    fp32_path = directory / f"matmul-{k}x{n}-fp32.onnx"
    qint8_path = directory / f"matmul-{k}x{n}-qint8.onnx"
    onnx.save(model, fp32_path)
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(qint8_path),
        per_channel=True,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        extra_options={
            "WeightSymmetric": True,
            "MatMulConstBOnly": True,
        },
    )
    return fp32_path, qint8_path


def _run_once(
    session: Any,
    feeds: Mapping[str, NDArray[Any]],
) -> tuple[float, float]:
    wall_started = perf_counter()
    cpu_started = process_time()
    session.run(None, feeds)
    cpu_ms = (process_time() - cpu_started) * 1_000.0
    wall_ms = (perf_counter() - wall_started) * 1_000.0
    return wall_ms, cpu_ms


def _latency_summary(
    samples_ms: Sequence[float],
    *,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, float]:
    values = np.asarray(samples_ms, dtype=np.float64)
    median = float(np.median(values))
    confidence = (
        bootstrap_median_ci(samples_ms, seed=seed, resamples=bootstrap_resamples)
        if median > 0.0
        else {
            "median_ci95_low_ms": 0.0,
            "median_ci95_high_ms": 0.0,
            "median_ci95_half_width_percent": 0.0,
        }
    )
    return {
        "median_ms": median,
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(np.mean(values)),
        "stdev_ms": float(np.std(values)),
        "minimum_ms": float(np.min(values)),
        "maximum_ms": float(np.max(values)),
        **confidence,
    }


def _measure_pair(
    sessions: Mapping[str, Any],
    feeds: Mapping[str, NDArray[Any]],
    *,
    warmups: int,
    block_warmups: int,
    iterations: int,
    measurement_blocks: int,
    seed: int,
) -> tuple[dict[str, list[float]], dict[str, list[float]], list[dict[str, Any]]]:
    names = list(sessions)
    rng = random.Random(seed)
    for _ in range(warmups):
        order = names.copy()
        rng.shuffle(order)
        for name in order:
            sessions[name].run(None, feeds)

    wall_samples = {name: [] for name in names}
    cpu_samples = {name: [] for name in names}
    orders: list[list[str]] = []
    for block_index in range(measurement_blocks):
        if block_index % 2 == 0:
            order = names.copy()
            rng.shuffle(order)
        else:
            order = list(reversed(orders[-1]))
        orders.append(order)

    blocks: list[dict[str, Any]] = []
    for block_index, (count, order) in enumerate(
        zip(_trial_sizes(iterations, measurement_blocks), orders, strict=True)
    ):
        block_wall: dict[str, list[float]] = {}
        block_cpu: dict[str, list[float]] = {}
        for name in order:
            for _ in range(block_warmups):
                sessions[name].run(None, feeds)
            current_wall: list[float] = []
            current_cpu: list[float] = []
            for _ in range(count):
                wall_ms, cpu_ms = _run_once(sessions[name], feeds)
                current_wall.append(wall_ms)
                current_cpu.append(cpu_ms)
            block_wall[name] = current_wall
            block_cpu[name] = current_cpu
            wall_samples[name].extend(current_wall)
            cpu_samples[name].extend(current_cpu)
        blocks.append(
            {
                "block_index": block_index,
                "order": order,
                "discarded_warmups_per_precision": block_warmups,
                "iterations_per_precision": count,
                "samples_ms": block_wall,
                "process_cpu_samples_ms": block_cpu,
            }
        )
    return wall_samples, cpu_samples, blocks


def measure_exact_shape_kernels(
    baseline_model_path: Path,
    *,
    rows: Sequence[int],
    threads: int,
    warmups: int,
    block_warmups: int,
    iterations: int,
    measurement_blocks: int,
    random_seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """Measure isolated FP32 and dynamic-QInt8 MatMuls for every target shape."""

    if not rows or any(value < 1 for value in rows):
        raise ValueError("rows must contain positive integers")
    if len(set(rows)) != len(rows):
        raise ValueError("rows must not contain duplicates")
    if threads < 1 or warmups < 0 or block_warmups < 0:
        raise ValueError("threads must be positive and warmups must be non-negative")
    _trial_sizes(iterations, measurement_blocks)
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")

    baseline_model_path = baseline_model_path.resolve()
    shape_counts = _target_shape_counts(baseline_model_path)
    results: list[dict[str, Any]] = []
    session_loads: list[dict[str, float | int]] = []
    with TemporaryDirectory(prefix="armbench-kernels-") as temporary:
        directory = Path(temporary)
        for shape_index, ((k, n), node_count) in enumerate(sorted(shape_counts.items())):
            fp32_path, qint8_path = _write_matmul_pair(
                directory,
                k=k,
                n=n,
                seed=random_seed + shape_index * 10_007,
            )
            fp32_session, fp32_load_ms = _create_session(fp32_path, threads=threads)
            qint8_session, qint8_load_ms = _create_session(qint8_path, threads=threads)
            session_loads.append(
                {"k": k, "n": n, "fp32_ms": fp32_load_ms, "qint8_ms": qint8_load_ms}
            )
            sessions = {"fp32": fp32_session, "qint8": qint8_session}
            for row_index, row_count in enumerate(rows):
                input_rng = np.random.default_rng(
                    random_seed + shape_index * 10_007 + row_index * 1_009 + 1
                )
                feeds = {
                    "input": input_rng.uniform(-1.0, 1.0, size=(row_count, k)).astype(
                        np.float32
                    )
                }
                case_seed = random_seed + shape_index * 100_003 + row_index * 1_009
                wall, cpu, blocks = _measure_pair(
                    sessions,
                    feeds,
                    warmups=warmups,
                    block_warmups=block_warmups,
                    iterations=iterations,
                    measurement_blocks=measurement_blocks,
                    seed=case_seed,
                )
                flops = 2 * row_count * k * n
                precision_results: dict[str, Any] = {}
                for precision, offset in (("fp32", 1), ("qint8", 2)):
                    wall_summary = _latency_summary(
                        wall[precision],
                        seed=case_seed + offset,
                        bootstrap_resamples=bootstrap_resamples,
                    )
                    cpu_summary = _latency_summary(
                        cpu[precision],
                        seed=case_seed + offset + 2,
                        bootstrap_resamples=bootstrap_resamples,
                    )
                    wall_summary["effective_gigaops_per_second"] = flops / (
                        wall_summary["median_ms"] * 1_000_000.0
                    )
                    precision_results[precision] = {
                        "wall": wall_summary,
                        "process_cpu": cpu_summary,
                    }
                results.append(
                    {
                        "rows": row_count,
                        "k": k,
                        "n": n,
                        "target_node_count": node_count,
                        "flops_per_call": flops,
                        "random_seed": case_seed,
                        "fp32": precision_results["fp32"],
                        "qint8": precision_results["qint8"],
                        "median_latency_speedup": precision_results["fp32"]["wall"][
                            "median_ms"
                        ]
                        / precision_results["qint8"]["wall"]["median_ms"],
                        "measurement_blocks": blocks,
                    }
                )
            del sessions, fp32_session, qint8_session

    return {
        "schema_version": 1,
        "benchmark": "ONNX Runtime exact-shape constant-weight MatMul kernels",
        "baseline_model": {
            "filename": baseline_model_path.name,
            "sha256": _sha256(baseline_model_path),
        },
        "target_weight_shapes": [
            {"k": k, "n": n, "target_node_count": count}
            for (k, n), count in sorted(shape_counts.items())
        ],
        "configuration": {
            "rows": list(rows),
            "warmups_per_precision_and_case": warmups,
            "discarded_warmups_per_precision_and_block": block_warmups,
            "measured_iterations_per_precision_and_case": iterations,
            "measurement_blocks_per_case": measurement_blocks,
            "random_seed": random_seed,
            "bootstrap_resamples": bootstrap_resamples,
            "intra_op_threads": threads,
            "inter_op_threads": 1,
            "intra_op_spin_duration_us": INTRA_OP_SPIN_DURATION_US,
            "intra_op_spin_backoff_max": INTRA_OP_SPIN_BACKOFF_MAX,
            "graph_optimization_level": "ORT_ENABLE_ALL",
            "execution_mode": "ORT_SEQUENTIAL",
        },
        "machine": _machine_metadata(),
        "session_load_ms_by_shape": session_loads,
        "results": results,
        "notes": [
            "Each graph contains one MatMul with a dynamic row dimension and a constant weight.",
            (
                "QInt8 uses the same dynamic, per-channel, signed weight quantization "
                "configuration as the full model."
            ),
            (
                "Repeated isolated calls are a hot-weight best case, not a hardware peak or "
                "full-model latency prediction."
            ),
            (
                "Equivalent QInt8 gigaops use the FP32 operation count (two operations per "
                "multiply-accumulate)."
            ),
        ],
    }


def write_kernel_microbenchmark(result: Mapping[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output
