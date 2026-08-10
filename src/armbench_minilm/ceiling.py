"""Operator-scope Amdahl analysis and preliminary roofline accounting."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import onnx

_KERNEL_SUFFIX = "_kernel_time"
_QUANTIZED_TARGET_OPERATORS = {
    "DynamicQuantizeLinear",
    "DynamicQuantizeMatMul",
    "MatMulIntegerToFloat",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _elements(dimensions: Sequence[int]) -> int:
    return math.prod(dimensions)


def _target_nodes(model_path: Path) -> dict[str, dict[str, Any]]:
    model = onnx.load(model_path.resolve(), load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    result: dict[str, dict[str, Any]] = {}
    for node in model.graph.node:
        if (
            node.op_type not in {"MatMul", "Gemm"}
            or len(node.input) < 2
            or node.input[1] not in initializers
        ):
            continue
        if not node.name:
            raise ValueError("target MatMul/Gemm node has no stable name")
        tensor = initializers[node.input[1]]
        dimensions = [int(value) for value in tensor.dims]
        if len(dimensions) != 2:
            raise ValueError(f"target weight is not a matrix: {node.name}: {dimensions}")
        result[node.name] = {
            "operator": node.op_type,
            "weight_name": tensor.name,
            "weight_dimensions": dimensions,
            "weight_elements": _elements(dimensions),
            "weight_bytes_fp32": _elements(dimensions) * 4,
        }
    if not result:
        raise ValueError("baseline graph has no constant-weight MatMul/Gemm target nodes")
    return result


def _profile_node_name(event_name: str) -> str:
    if event_name.endswith(_KERNEL_SUFFIX):
        return event_name[: -len(_KERNEL_SUFFIX)]
    return event_name


def _shape_list(value: Any) -> list[list[int]]:
    entries = value if isinstance(value, list) else [value]
    shapes: list[list[int]] = []
    for entry in entries:
        if not isinstance(entry, Mapping) or len(entry) != 1:
            continue
        dimensions = next(iter(entry.values()))
        if isinstance(dimensions, list) and all(isinstance(item, int) for item in dimensions):
            shapes.append(dimensions)
    return shapes


def _raw_profile_nodes(path: Path) -> list[dict[str, Any]]:
    events = json.loads(path.read_text(encoding="utf-8"))
    groups: dict[str, dict[str, Any]] = {}
    for event in events:
        args = event.get("args", {})
        duration = event.get("dur")
        event_name = event.get("name")
        operator = args.get("op_name")
        if (
            event.get("cat") != "Node"
            or not isinstance(duration, int | float)
            or not isinstance(event_name, str)
            or not isinstance(operator, str)
        ):
            continue
        node_name = _profile_node_name(event_name)
        if node_name not in groups:
            groups[node_name] = {
                "name": node_name,
                "operator": operator,
                "duration_us": 0.0,
                "calls": 0,
                "input_type_shape": args.get("input_type_shape"),
                "output_type_shape": args.get("output_type_shape"),
            }
        groups[node_name]["duration_us"] += float(duration)
        groups[node_name]["calls"] += 1
    nodes = list(groups.values())
    nodes.sort(key=lambda item: float(item["duration_us"]), reverse=True)
    return nodes


def _profile_nodes(summary: Mapping[str, Any], profile_dir: Path) -> list[dict[str, Any]]:
    embedded = summary.get("nodes")
    if isinstance(embedded, list) and embedded:
        return [dict(item) for item in embedded]
    profile_file = summary.get("profile_file")
    if not isinstance(profile_file, str):
        raise ValueError("profile summary has neither embedded nodes nor a profile_file")
    profile_path = profile_dir / profile_file
    if not profile_path.is_file():
        raise FileNotFoundError(f"raw ORT profile is required: {profile_path}")
    return _raw_profile_nodes(profile_path)


def _profiled_inferences(nodes: Sequence[Mapping[str, Any]]) -> int:
    call_counts = [int(item["calls"]) for item in nodes if int(item["calls"]) > 0]
    if not call_counts:
        raise ValueError("profile has no node calls")
    count = Counter(call_counts).most_common(1)[0][0]
    if count < 1:
        raise ValueError("could not infer the number of profiled inferences")
    return count


def _broadcast_dimensions(left: Sequence[int], right: Sequence[int]) -> list[int]:
    output: list[int] = []
    for left_value, right_value in zip(reversed(left), reversed(right), strict=False):
        if left_value == right_value or left_value == 1 or right_value == 1:
            output.append(max(left_value, right_value))
        else:
            raise ValueError(f"cannot broadcast MatMul batches: {left} and {right}")
    longer = left if len(left) > len(right) else right
    output.extend(reversed(longer[: abs(len(left) - len(right))]))
    return list(reversed(output))


def _dynamic_matmul_flops(node: Mapping[str, Any]) -> int:
    shapes = _shape_list(node.get("input_type_shape"))
    if len(shapes) != 2 or len(shapes[0]) < 2 or len(shapes[1]) < 2:
        raise ValueError(f"dynamic MatMul profile lacks two matrix shapes: {node['name']}")
    left, right = shapes
    if left[-1] != right[-2]:
        raise ValueError(f"MatMul dimensions do not align: {left} and {right}")
    batch = _elements(_broadcast_dimensions(left[:-2], right[:-2]))
    return 2 * batch * left[-2] * left[-1] * right[-1]


def _operator_totals(nodes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"duration_us": 0.0, "calls": 0}
    )
    for node in nodes:
        operator = str(node["operator"])
        totals[operator]["duration_us"] += float(node["duration_us"])
        totals[operator]["calls"] += int(node["calls"])
    total_duration = sum(float(item["duration_us"]) for item in totals.values())
    result = [
        {
            "operator": operator,
            "duration_us": values["duration_us"],
            "calls": values["calls"],
            "node_time_share": (
                float(values["duration_us"]) / total_duration if total_duration else 0.0
            ),
        }
        for operator, values in totals.items()
    ]
    result.sort(key=lambda item: float(item["duration_us"]), reverse=True)
    return result


def _memory_bandwidth_for_threads(
    microbenchmarks: Sequence[Mapping[str, Any]],
    *,
    threads: int,
) -> dict[str, Any] | None:
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for microbenchmark in microbenchmarks:
        matches = [
            item
            for item in microbenchmark.get("results", [])
            if int(item["threads"]) == threads
        ]
        if matches:
            selected.append(
                (microbenchmark, max(matches, key=lambda item: int(item["size_bytes"])))
            )
    if not selected:
        return None
    sizes = {int(item["size_bytes"]) for _, item in selected}
    if len(sizes) != 1:
        raise ValueError(f"memory microbenchmarks use different largest sizes: {sorted(sizes)}")
    values = [float(item["median_gbps"]) for _, item in selected]
    return {
        "size_bytes": sizes.pop(),
        "threads": threads,
        "median_gbps": statistics.median(values),
        "minimum_run_median_gbps": min(values),
        "maximum_run_median_gbps": max(values),
        "run_median_cv_percent": (
            statistics.pstdev(values) / statistics.mean(values) * 100.0
            if len(values) > 1
            else 0.0
        ),
        "runs": [
            {
                "github_run_id": benchmark.get("machine", {}).get("github_run_id"),
                "median_gbps": float(item["median_gbps"]),
            }
            for benchmark, item in selected
        ],
    }


def _aggregate_metric(values: Sequence[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "cv_percent": (
            statistics.pstdev(values) / statistics.mean(values) * 100.0
            if len(values) > 1
            else 0.0
        ),
    }


def _aggregate_kernel_microbenchmarks(
    microbenchmarks: Sequence[Mapping[str, Any]],
    *,
    model_hash: str,
    target_nodes: Mapping[str, Mapping[str, Any]],
    threads: int,
    expected_run_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    if not microbenchmarks:
        return None
    expected_shapes = Counter(
        tuple(int(value) for value in item["weight_dimensions"])
        for item in target_nodes.values()
    )
    grouped: dict[tuple[int, int, int], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = (
        defaultdict(list)
    )
    run_ids = {
        str(benchmark.get("machine", {}).get("github_run_id"))
        for benchmark in microbenchmarks
        if benchmark.get("machine", {}).get("github_run_id") is not None
    }
    if expected_run_ids and run_ids != expected_run_ids:
        raise ValueError(
            f"kernel microbenchmark run IDs do not match evidence: {run_ids} != "
            f"{expected_run_ids}"
        )
    for benchmark in microbenchmarks:
        current_hash = str(benchmark.get("baseline_model", {}).get("sha256"))
        if current_hash != model_hash:
            raise ValueError(
                "kernel microbenchmark baseline hash does not match analyzed model: "
                f"{current_hash} != {model_hash}"
            )
        current_threads = int(benchmark["configuration"]["intra_op_threads"])
        if current_threads != threads:
            raise ValueError(
                f"kernel microbenchmark uses {current_threads} threads, expected {threads}"
            )
        actual_shapes = Counter(
            {
                (int(item["k"]), int(item["n"])): int(item["target_node_count"])
                for item in benchmark["target_weight_shapes"]
            }
        )
        if actual_shapes != expected_shapes:
            raise ValueError(
                f"kernel target shapes do not match model: {actual_shapes} != {expected_shapes}"
            )
        for item in benchmark["results"]:
            key = (int(item["rows"]), int(item["k"]), int(item["n"]))
            grouped[key].append((benchmark, item))

    cases: list[dict[str, Any]] = []
    for (rows, k, n), entries in sorted(grouped.items()):
        if len(entries) != len(microbenchmarks):
            raise ValueError(f"kernel microbenchmark is missing repeated case {(rows, k, n)}")
        count = expected_shapes.get((k, n))
        if count is None:
            raise ValueError(f"kernel microbenchmark contains unexpected shape {(k, n)}")
        flops = 2 * rows * k * n
        case: dict[str, Any] = {
            "rows": rows,
            "k": k,
            "n": n,
            "target_node_count": count,
            "flops_per_call": flops,
            "runs": len(entries),
        }
        for precision in ("fp32", "qint8"):
            latencies = [
                float(item[precision]["wall"]["median_ms"])
                for _, item in entries
            ]
            latency = _aggregate_metric(latencies)
            latency["effective_gigaops_per_second"] = flops / (
                latency["median"] * 1_000_000.0
            )
            case[precision] = latency
        case["median_latency_speedup"] = (
            case["fp32"]["median"] / case["qint8"]["median"]
        )
        case["run_ids"] = [
            benchmark.get("machine", {}).get("github_run_id")
            for benchmark, _ in entries
        ]
        cases.append(case)

    rows_present = {int(item["rows"]) for item in cases}
    for rows in rows_present:
        shapes = Counter(
            {
                (int(item["k"]), int(item["n"])): int(item["target_node_count"])
                for item in cases
                if int(item["rows"]) == rows
            }
        )
        if shapes != expected_shapes:
            raise ValueError(f"kernel row {rows} does not cover all target shapes")

    peak: dict[str, Any] = {}
    for precision in ("fp32", "qint8"):
        best = max(cases, key=lambda item: float(item[precision]["effective_gigaops_per_second"]))
        peak[precision] = {
            "effective_gigaops_per_second": best[precision][
                "effective_gigaops_per_second"
            ],
            "rows": best["rows"],
            "k": best["k"],
            "n": best["n"],
            "interpretation": "Best median-of-run-medians from an isolated hot-weight graph.",
        }
    return {
        "runs": len(microbenchmarks),
        "threads": threads,
        "peak": peak,
        "maximum_run_median_cv_percent": max(
            float(item[precision]["cv_percent"])
            for item in cases
            for precision in ("fp32", "qint8")
        ),
        "cases": cases,
    }


def _kernel_reference_for_rows(
    kernel_ceiling: Mapping[str, Any],
    *,
    rows: int,
) -> dict[str, Any]:
    cases = [item for item in kernel_ceiling["cases"] if int(item["rows"]) == rows]
    if not cases:
        raise ValueError(f"kernel microbenchmark has no row count {rows}")
    result: dict[str, Any] = {"rows": rows, "shapes": len(cases)}
    total_flops = sum(
        int(item["target_node_count"]) * int(item["flops_per_call"])
        for item in cases
    )
    result["target_equivalent_ops"] = total_flops
    for precision in ("fp32", "qint8"):
        latency_ms = sum(
            int(item["target_node_count"]) * float(item[precision]["median"])
            for item in cases
        )
        result[precision] = {
            "isolated_target_time_ms": latency_ms,
            "effective_gigaops_per_second": total_flops / (latency_ms * 1_000_000.0),
        }
    result["isolated_target_speedup"] = (
        result["fp32"]["isolated_target_time_ms"]
        / result["qint8"]["isolated_target_time_ms"]
    )
    return result


def _case_analysis(
    benchmark: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    profile_dir: Path,
    target_nodes: Mapping[str, Mapping[str, Any]],
    memory_bandwidth: Mapping[str, Any] | None,
    kernel_ceiling: Mapping[str, Any] | None,
) -> dict[str, Any]:
    batch_size = int(profile["batch_size"])
    sequence_length = int(profile["sequence_length"])
    measured = next(
        item
        for item in benchmark["batches"]
        if int(item["batch_size"]) == batch_size
        and int(item["sequence_length"]) == sequence_length
    )

    baseline_nodes = _profile_nodes(profile["baseline"], profile_dir)
    optimized_nodes = _profile_nodes(profile["optimized"], profile_dir)
    baseline_calls = _profiled_inferences(baseline_nodes)
    optimized_calls = _profiled_inferences(optimized_nodes)
    baseline_by_name = {str(item["name"]): item for item in baseline_nodes}
    missing = sorted(set(target_nodes).difference(baseline_by_name))
    if missing:
        raise ValueError(f"ORT profile is missing {len(missing)} target nodes: {missing[:3]}")

    total_duration_us = sum(float(item["duration_us"]) for item in baseline_nodes)
    target_duration_us = sum(
        float(baseline_by_name[name]["duration_us"]) for name in target_nodes
    )
    target_share = target_duration_us / total_duration_us
    infinite_speedup = 1.0 / (1.0 - target_share)

    optimized_target_duration_us = sum(
        float(item["duration_us"])
        for item in optimized_nodes
        if item["operator"] in _QUANTIZED_TARGET_OPERATORS
    )
    baseline_target_ms = target_duration_us / baseline_calls / 1_000.0
    optimized_target_ms = optimized_target_duration_us / optimized_calls / 1_000.0
    target_pipeline_speedup = baseline_target_ms / optimized_target_ms
    finite_prediction = 1.0 / (
        (1.0 - target_share) + target_share / target_pipeline_speedup
    )

    rows = batch_size * sequence_length
    target_flops = sum(
        2 * rows * int(item["weight_elements"]) for item in target_nodes.values()
    )
    target_logical_bytes = sum(
        int(item["weight_bytes_fp32"])
        + 4 * rows * int(item["weight_dimensions"][0])
        + 4 * rows * int(item["weight_dimensions"][1])
        for item in target_nodes.values()
    )
    target_qint8_logical_bytes = sum(
        int(item["weight_elements"])
        + 4 * rows * int(item["weight_dimensions"][0])
        + 4 * rows * int(item["weight_dimensions"][1])
        for item in target_nodes.values()
    )
    dynamic_matmul_flops = sum(
        _dynamic_matmul_flops(item)
        for item in baseline_nodes
        if item["operator"] == "MatMul" and item["name"] not in target_nodes
    )
    observed_speedup = float(measured["median_latency_speedup"])
    result: dict[str, Any] = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "profiled_inferences": {
            "baseline": baseline_calls,
            "optimized": optimized_calls,
        },
        "baseline_node_time_ms_per_inference": total_duration_us
        / baseline_calls
        / 1_000.0,
        "baseline_target_time_ms_per_inference": baseline_target_ms,
        "baseline_target_node_time_share": target_share,
        "operator_scope_infinite_speedup_limit": infinite_speedup,
        "observed_end_to_end_speedup": observed_speedup,
        "infinite_potential_realized_percent": (
            (observed_speedup - 1.0) / (infinite_speedup - 1.0) * 100.0
        ),
        "optimized_target_pipeline_time_ms_per_inference": optimized_target_ms,
        "profile_target_pipeline_speedup": target_pipeline_speedup,
        "amdahl_speedup_at_profiled_target_rate": finite_prediction,
        "target_fp32_flops": target_flops,
        "dynamic_attention_matmul_flops": dynamic_matmul_flops,
        "total_matmul_flops": target_flops + dynamic_matmul_flops,
        "target_share_of_matmul_flops": target_flops
        / (target_flops + dynamic_matmul_flops),
        "target_fp32_logical_bytes": target_logical_bytes,
        "target_qint8_minimum_logical_bytes": target_qint8_logical_bytes,
        "target_fp32_arithmetic_intensity_flops_per_logical_byte": target_flops
        / target_logical_bytes,
        "baseline_target_effective_gflops": target_flops / (baseline_target_ms * 1_000_000.0),
        "optimized_target_effective_equivalent_gops": target_flops
        / (optimized_target_ms * 1_000_000.0),
        "baseline_top_operators": _operator_totals(baseline_nodes)[:8],
        "optimized_top_operators": _operator_totals(optimized_nodes)[:8],
    }
    if memory_bandwidth is not None:
        bandwidth_gbps = float(memory_bandwidth["median_gbps"])
        result["memory_projection"] = {
            "source_size_bytes": int(memory_bandwidth["size_bytes"]),
            "threads": int(memory_bandwidth["threads"]),
            "median_gbps": bandwidth_gbps,
            "run_median_cv_percent": float(
                memory_bandwidth["run_median_cv_percent"]
            ),
            "runs": memory_bandwidth["runs"],
            "target_logical_traffic_time_ms": target_logical_bytes
            / (bandwidth_gbps * 1_000_000_000.0)
            * 1_000.0,
            "interpretation": (
                "Projection at measured copy bandwidth; not a cache-aware traffic bound."
            ),
        }
    if kernel_ceiling is not None:
        reference = _kernel_reference_for_rows(kernel_ceiling, rows=rows)
        fp32_reference_ms = float(reference["fp32"]["isolated_target_time_ms"])
        qint8_reference_ms = float(reference["qint8"]["isolated_target_time_ms"])
        target_speedup = baseline_target_ms / qint8_reference_ms
        reference["baseline_profile_to_isolated_fp32_time_ratio"] = (
            baseline_target_ms / fp32_reference_ms
        )
        reference["optimized_profile_to_isolated_qint8_time_ratio"] = (
            optimized_target_ms / qint8_reference_ms
        )
        reference["qint8_profile_minus_isolated_time_ms"] = (
            optimized_target_ms - qint8_reference_ms
        )
        reference["amdahl_speedup_at_isolated_qint8_target_time"] = 1.0 / (
            (1.0 - target_share) + target_share / target_speedup
        )
        fp32_peak = float(
            kernel_ceiling["peak"]["fp32"]["effective_gigaops_per_second"]
        )
        qint8_peak = float(
            kernel_ceiling["peak"]["qint8"]["effective_gigaops_per_second"]
        )
        reference["compute_projection"] = {
            "fp32_peak_effective_gflops": fp32_peak,
            "qint8_peak_effective_equivalent_gops": qint8_peak,
            "fp32_target_time_ms": target_flops / (fp32_peak * 1_000_000.0),
            "qint8_target_time_ms": target_flops / (qint8_peak * 1_000_000.0),
        }
        if memory_bandwidth is not None:
            bandwidth_gbps = float(memory_bandwidth["median_gbps"])
            fp32_memory_ms = target_logical_bytes / (bandwidth_gbps * 1_000_000.0)
            qint8_memory_ms = target_qint8_logical_bytes / (
                bandwidth_gbps * 1_000_000.0
            )
            reference["preliminary_roofline"] = {
                "fp32_minimum_time_ms": max(
                    reference["compute_projection"]["fp32_target_time_ms"],
                    fp32_memory_ms,
                ),
                "qint8_minimum_time_ms": max(
                    reference["compute_projection"]["qint8_target_time_ms"],
                    qint8_memory_ms,
                ),
                "fp32_compute_time_ms": reference["compute_projection"][
                    "fp32_target_time_ms"
                ],
                "qint8_compute_time_ms": reference["compute_projection"][
                    "qint8_target_time_ms"
                ],
                "fp32_logical_traffic_time_ms": fp32_memory_ms,
                "qint8_minimum_logical_traffic_time_ms": qint8_memory_ms,
                "interpretation": (
                    "Uses the best isolated shape rate and copy bandwidth; cache traffic and "
                    "dynamic-quantization scratch traffic remain unmeasured."
                ),
            }
        result["exact_shape_kernel_reference"] = reference
        result["isolated_fp32_target_time_ms"] = fp32_reference_ms
        result["isolated_qint8_target_time_ms"] = qint8_reference_ms
        result["isolated_target_speedup"] = reference["isolated_target_speedup"]
        result["amdahl_speedup_at_isolated_qint8_target_time"] = reference[
            "amdahl_speedup_at_isolated_qint8_target_time"
        ]
        result["optimized_profile_to_isolated_qint8_time_ratio"] = reference[
            "optimized_profile_to_isolated_qint8_time_ratio"
        ]
    return result


def _aggregate_cases(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(
        {
            (int(case["batch_size"]), int(case["sequence_length"]))
            for run in runs
            for case in run["cases"]
        }
    )
    fields = (
        "baseline_target_node_time_share",
        "operator_scope_infinite_speedup_limit",
        "observed_end_to_end_speedup",
        "infinite_potential_realized_percent",
        "profile_target_pipeline_speedup",
        "amdahl_speedup_at_profiled_target_rate",
        "baseline_target_effective_gflops",
        "optimized_target_effective_equivalent_gops",
    )
    result: list[dict[str, Any]] = []
    for batch_size, sequence_length in keys:
        cases = [
            case
            for run in runs
            for case in run["cases"]
            if int(case["batch_size"]) == batch_size
            and int(case["sequence_length"]) == sequence_length
        ]
        aggregate: dict[str, Any] = {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "runs": len(cases),
            "target_fp32_flops": cases[0]["target_fp32_flops"],
            "dynamic_attention_matmul_flops": cases[0]["dynamic_attention_matmul_flops"],
            "target_fp32_logical_bytes": cases[0]["target_fp32_logical_bytes"],
            "target_qint8_minimum_logical_bytes": cases[0][
                "target_qint8_minimum_logical_bytes"
            ],
            "target_fp32_arithmetic_intensity_flops_per_logical_byte": cases[0][
                "target_fp32_arithmetic_intensity_flops_per_logical_byte"
            ],
        }
        for field in fields:
            values = [float(case[field]) for case in cases]
            aggregate[field] = {
                "mean": statistics.mean(values),
                "minimum": min(values),
                "maximum": max(values),
                "cv_percent": (
                    statistics.pstdev(values) / statistics.mean(values) * 100.0
                    if len(values) > 1
                    else 0.0
                ),
            }
        optional_fields = (
            "isolated_fp32_target_time_ms",
            "isolated_qint8_target_time_ms",
            "isolated_target_speedup",
            "amdahl_speedup_at_isolated_qint8_target_time",
            "optimized_profile_to_isolated_qint8_time_ratio",
        )
        for field in optional_fields:
            if field not in cases[0]:
                continue
            values = [float(case[field]) for case in cases]
            aggregate[field] = {
                "mean": statistics.mean(values),
                "minimum": min(values),
                "maximum": max(values),
                "cv_percent": (
                    statistics.pstdev(values) / statistics.mean(values) * 100.0
                    if len(values) > 1
                    else 0.0
                ),
            }
        result.append(aggregate)
    return result


def analyze_ceiling_runs(
    baseline_model_path: Path,
    evidence_dirs: Sequence[Path],
    *,
    memory_microbenchmark_paths: Sequence[Path] | None = None,
    kernel_microbenchmark_paths: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Analyze exact constant-weight target nodes across one or more benchmark runs."""

    if not evidence_dirs:
        raise ValueError("at least one evidence directory is required")
    baseline_model_path = baseline_model_path.resolve()
    model_hash = _sha256(baseline_model_path)
    target_nodes = _target_nodes(baseline_model_path)
    microbenchmarks = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in memory_microbenchmark_paths or []
    ]
    kernel_microbenchmarks = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in kernel_microbenchmark_paths or []
    ]
    evidence: list[tuple[Path, dict[str, Any]]] = []
    for evidence_dir in evidence_dirs:
        evidence_dir = evidence_dir.resolve()
        benchmark_path = evidence_dir / "benchmark.json"
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        current_hash = str(benchmark["models"]["baseline"]["sha256"])
        if current_hash != model_hash:
            raise ValueError(
                f"evidence baseline hash does not match {baseline_model_path.name}: "
                f"{current_hash} != {model_hash}"
            )
        evidence.append((evidence_dir, benchmark))
    configured_threads = {
        int(benchmark["configuration"]["intra_op_threads"])
        for _, benchmark in evidence
    }
    if len(configured_threads) != 1:
        raise ValueError(f"evidence runs use different thread counts: {configured_threads}")
    threads = configured_threads.pop()
    expected_run_ids = {
        str(benchmark["machine"]["github_run_id"])
        for _, benchmark in evidence
        if benchmark["machine"].get("github_run_id") is not None
    }
    kernel_ceiling = _aggregate_kernel_microbenchmarks(
        kernel_microbenchmarks,
        model_hash=model_hash,
        target_nodes=target_nodes,
        threads=threads,
        expected_run_ids=expected_run_ids,
    )

    runs: list[dict[str, Any]] = []
    for evidence_dir, benchmark in evidence:
        bandwidth = _memory_bandwidth_for_threads(microbenchmarks, threads=threads)
        cases = [
            _case_analysis(
                benchmark,
                profile,
                profile_dir=evidence_dir / "profiles",
                target_nodes=target_nodes,
                memory_bandwidth=bandwidth,
                kernel_ceiling=kernel_ceiling,
            )
            for profile in benchmark["profiles"]
        ]
        runs.append(
            {
                "evidence_id": benchmark["machine"].get("github_run_id")
                or evidence_dir.name,
                "github_run_id": benchmark["machine"].get("github_run_id"),
                "github_sha": benchmark["machine"].get("github_sha"),
                "cpu_part": (benchmark["machine"].get("linux_cpu") or {})
                .get("identity", {})
                .get("cpu_part"),
                "runner_image_version": benchmark["machine"].get(
                    "github_runner_image_version"
                ),
                "cases": cases,
            }
        )
    target_elements = sum(int(item["weight_elements"]) for item in target_nodes.values())
    return {
        "schema_version": 2,
        "analysis": "operator-scope Amdahl bound and measured exact-shape roofline accounting",
        "baseline_model": {
            "path": baseline_model_path.name,
            "sha256": model_hash,
        },
        "target_scope": {
            "operator_types": ["Gemm", "MatMul"],
            "constant_weight_nodes": len(target_nodes),
            "float32_weight_elements": target_elements,
            "float32_weight_bytes": target_elements * 4,
        },
        "roofline_status": {
            "complete": False,
            "available": [
                "exact target and dynamic-attention MatMul FLOPs",
                "logical FP32 target bytes",
                "measured operator time",
                "optional measured copy bandwidth",
                *(
                    [
                        "independent hot-weight FP32 and dynamic-QInt8 rates for every exact "
                        "target shape"
                    ]
                    if kernel_ceiling is not None
                    else []
                ),
            ],
            "missing": [
                *(
                    ["independent FP32 and INT8 compute ceilings for the exact matrix shapes"]
                    if kernel_ceiling is None
                    else []
                ),
                "cache-hierarchy traffic or hardware-counter measurements",
            ],
        },
        "kernel_ceiling": kernel_ceiling,
        "runs": runs,
        "aggregate": {"cases": _aggregate_cases(runs)},
        "notes": [
            (
                "The infinite-speedup limit uses the target share of profiled ORT node time, "
                "so it is an operator-scope ceiling rather than a hardware peak claim."
            ),
            (
                "Logical bytes count each target weight, input, and output once per node; cache "
                "reuse and actual memory-controller traffic require separate measurement."
            ),
            (
                "Exact-shape kernels repeatedly reuse one synthetic weight per shape, making them "
                "an optimistic hot-weight reference rather than a physical hardware peak."
            ),
            "ORT profiling runs in separate sessions and does not alter timed latency samples.",
        ],
    }


def write_ceiling_reports(result: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ceiling.json"
    markdown_path = output_dir / "ceiling.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# ArmBench operator-scope ceiling analysis",
        "",
        (
            f"Exact scope: **{result['target_scope']['constant_weight_nodes']}** constant-weight "
            f"MatMul/Gemm nodes and {result['target_scope']['float32_weight_bytes'] / 2**20:.3f} "
            "MiB of FP32 weights."
        ),
        "",
        (
            "| Batch | Sequence | Target node-time share | Infinite Amdahl limit | "
            "Observed speedup | Potential realized |"
        ),
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for case in result["aggregate"]["cases"]:
        share = case["baseline_target_node_time_share"]["mean"]
        limit = case["operator_scope_infinite_speedup_limit"]["mean"]
        observed = case["observed_end_to_end_speedup"]["mean"]
        realized = case["infinite_potential_realized_percent"]["mean"]
        lines.append(
            f"| {case['batch_size']} | {case['sequence_length']} | {share:.1%} | "
            f"{limit:.2f}x | {observed:.2f}x | {realized:.1f}% |"
        )
    kernel_ceiling = result.get("kernel_ceiling")
    if kernel_ceiling is not None:
        lines.extend(
            [
                "",
                "## Exact-shape kernel reference",
                "",
                (
                    f"Repeated runs: **{kernel_ceiling['runs']}**; maximum run-median CV: "
                    f"**{kernel_ceiling['maximum_run_median_cv_percent']:.2f}%**."
                ),
                "",
                (
                    "| Batch | Sequence | Rows | FP32 isolated target | QInt8 isolated target | "
                    "Isolated speedup | Profiled QInt8 / isolated | Projected Amdahl |"
                ),
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for case in result["aggregate"]["cases"]:
            if "isolated_fp32_target_time_ms" not in case:
                continue
            rows = int(case["batch_size"]) * int(case["sequence_length"])
            lines.append(
                f"| {case['batch_size']} | {case['sequence_length']} | {rows} | "
                f"{case['isolated_fp32_target_time_ms']['mean']:.3f} ms | "
                f"{case['isolated_qint8_target_time_ms']['mean']:.3f} ms | "
                f"{case['isolated_target_speedup']['mean']:.2f}x | "
                f"{case['optimized_profile_to_isolated_qint8_time_ratio']['mean']:.2f}x | "
                f"{case['amdahl_speedup_at_isolated_qint8_target_time']['mean']:.2f}x |"
            )
        lines.extend(
            [
                "",
                (
                    "Peak observed isolated rates: "
                    f"**{kernel_ceiling['peak']['fp32']['effective_gigaops_per_second']:.1f} "
                    "FP32 GFLOP/s** and "
                    f"**{kernel_ceiling['peak']['qint8']['effective_gigaops_per_second']:.1f} "
                    "equivalent QInt8 GOP/s**."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Roofline status",
            "",
            "This is deliberately **preliminary**, not a completed hardware roofline.",
            "",
            "Available:",
            "",
            *[f"- {item}" for item in result["roofline_status"]["available"]],
            "",
            "Still required:",
            "",
            *[f"- {item}" for item in result["roofline_status"]["missing"]],
            "",
            "## Interpretation limits",
            "",
            *[f"- {item}" for item in result["notes"]],
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
