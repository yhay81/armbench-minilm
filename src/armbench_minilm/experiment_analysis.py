"""Aggregate independent BF16 experiment runs into a promotion decision."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from armbench_minilm.constants import BF16_EXPERIMENT_ID

COMPARISON_NAMES = ("fp32_bf16_vs_control", "qint8_bf16_vs_control")
QUALITY_VARIANTS = ("fp32_bf16_fastmath", "qint8_bf16_fastmath")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evidence_file(path: Path) -> Path:
    candidate = path / "experiment.json" if path.is_dir() else path
    if not candidate.is_file():
        raise FileNotFoundError(f"BF16 experiment evidence not found: {candidate}")
    return candidate.resolve()


def load_bf16_experiment(path: Path) -> tuple[Path, dict[str, Any]]:
    """Load one experiment file from a direct path or artifact directory."""

    evidence_file = _evidence_file(path)
    result = json.loads(evidence_file.read_text(encoding="utf-8"))
    if result.get("schema_version") != "armbench-experiment/v1":
        raise ValueError(f"unsupported experiment schema: {evidence_file}")
    if result.get("experiment", {}).get("id") != BF16_EXPERIMENT_ID:
        raise ValueError(f"unexpected experiment ID: {evidence_file}")
    return evidence_file, result


def _aggregate(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot aggregate an empty metric")
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "minimum": min(values),
        "median": float(np.median(np.asarray(values, dtype=np.float64))),
        "mean": mean,
        "maximum": max(values),
        "cv_percent": statistics.pstdev(values) / mean * 100.0 if len(values) > 1 else 0.0,
    }


def _contract_signature(result: Mapping[str, Any]) -> dict[str, Any]:
    machine = result["machine"]
    cpu_identity = (machine.get("linux_cpu") or {}).get("identity", {})
    return {
        "experiment_id": result["experiment"]["id"],
        "parent": result["experiment"]["parent"],
        "source_model": result["source_model"],
        "configuration": result["configuration"],
        "variants": {
            name: {
                "model_kind": variant["model_kind"],
                "session_config": variant["session_config"],
                "sha256": variant["sha256"],
            }
            for name, variant in result["variants"].items()
        },
        "machine_contract": {
            "architecture": machine["architecture"],
            "onnxruntime": machine["onnxruntime"],
            "cpu_implementer": cpu_identity.get("cpu_implementer"),
            "cpu_part": cpu_identity.get("cpu_part"),
        },
    }


def _case_key(case: Mapping[str, Any]) -> tuple[int, int]:
    return int(case["batch_size"]), int(case["sequence_length"])


def _run_id(result: Mapping[str, Any]) -> str:
    run_id = result["machine"].get("github_run_id")
    if not run_id:
        raise ValueError("every promoted experiment must record machine.github_run_id")
    return str(run_id)


def _validate_runs(results: Sequence[Mapping[str, Any]]) -> None:
    if not results:
        raise ValueError("at least one BF16 experiment is required")
    expected_contract = _contract_signature(results[0])
    expected_shapes = {_case_key(case) for case in results[0]["cases"]}
    run_ids: set[str] = set()
    for result in results:
        run_id = _run_id(result)
        if run_id in run_ids:
            raise ValueError(f"duplicate GitHub run ID: {run_id}")
        run_ids.add(run_id)
        if _contract_signature(result) != expected_contract:
            raise ValueError(f"experiment contract mismatch in GitHub run {run_id}")
        shapes = {_case_key(case) for case in result["cases"]}
        if shapes != expected_shapes or len(shapes) != len(result["cases"]):
            raise ValueError(f"fixed-shape grid mismatch in GitHub run {run_id}")


def analyze_bf16_results(
    evidence: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    analysis_code_revision: str | None = None,
) -> dict[str, Any]:
    """Aggregate compatible native runs and apply the repeated-run gates."""

    results = [result for _, result in evidence]
    _validate_runs(results)
    first = results[0]
    shape_keys = sorted(_case_key(case) for case in first["cases"])
    cases_by_run = [{_case_key(case): case for case in result["cases"]} for result in results]

    run_summaries: list[dict[str, Any]] = []
    for (path, result), case_map in zip(evidence, cases_by_run, strict=True):
        comparisons = result["summary"]["comparisons"]
        run_summaries.append(
            {
                "github_run_id": _run_id(result),
                "github_run_url": (
                    f"https://github.com/yhay81/armbench-minilm/actions/runs/{_run_id(result)}"
                ),
                "code_revision": result["experiment"]["code_revision"],
                "evidence_sha256": _sha256(path),
                "recorded_verdict": result["verdict"]["status"],
                "comparisons": {name: comparisons[name] for name in COMPARISON_NAMES},
                "quality": {name: result["quality_by_variant"][name] for name in QUALITY_VARIANTS},
                "minimum_shape_speedups": {
                    name: min(
                        case_map[key]["comparisons"][name]["median_latency_speedup"]
                        for key in shape_keys
                    )
                    for name in COMPARISON_NAMES
                },
            }
        )

    comparison_aggregates: dict[str, Any] = {}
    per_shape: list[dict[str, Any]] = []
    for comparison_name in COMPARISON_NAMES:
        run_geometric_means = [
            float(run["comparisons"][comparison_name]["geometric_mean_latency_speedup"])
            for run in run_summaries
        ]
        comparison_aggregates[comparison_name] = {
            "run_geometric_mean_speedup": _aggregate(run_geometric_means),
            "minimum_speedup_across_all_runs_and_shapes": min(
                float(case_map[key]["comparisons"][comparison_name]["median_latency_speedup"])
                for case_map in cases_by_run
                for key in shape_keys
            ),
            "maximum_speedup_across_all_runs_and_shapes": max(
                float(case_map[key]["comparisons"][comparison_name]["median_latency_speedup"])
                for case_map in cases_by_run
                for key in shape_keys
            ),
        }

    for batch_size, sequence_length in shape_keys:
        item: dict[str, Any] = {
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "comparisons": {},
        }
        for comparison_name in COMPARISON_NAMES:
            values = [
                float(
                    case_map[(batch_size, sequence_length)]["comparisons"][comparison_name][
                        "median_latency_speedup"
                    ]
                )
                for case_map in cases_by_run
            ]
            item["comparisons"][comparison_name] = _aggregate(values)
        per_shape.append(item)

    quality_aggregates: dict[str, dict[str, Any]] = {
        variant_name: {
            "mean_embedding_cosine": _aggregate(
                [
                    float(result["quality_by_variant"][variant_name]["mean_embedding_cosine"])
                    for result in results
                ]
            ),
            "minimum_embedding_cosine_across_runs": min(
                float(result["quality_by_variant"][variant_name]["minimum_embedding_cosine"])
                for result in results
            ),
            "maximum_pairwise_similarity_error_across_runs": max(
                float(
                    result["quality_by_variant"][variant_name]["maximum_pairwise_similarity_error"]
                )
                for result in results
            ),
        }
        for variant_name in QUALITY_VARIANTS
    }

    fp32_summary = comparison_aggregates["fp32_bf16_vs_control"]
    fp32_per_shape_max_cv = max(
        item["comparisons"]["fp32_bf16_vs_control"]["cv_percent"] for item in per_shape
    )
    fp32_performance_gate = (
        len(results) >= 5
        and all(
            run["comparisons"]["fp32_bf16_vs_control"]["geometric_mean_latency_speedup"] >= 1.03
            for run in run_summaries
        )
        and fp32_summary["minimum_speedup_across_all_runs_and_shapes"] >= 0.97
        and fp32_summary["run_geometric_mean_speedup"]["cv_percent"] < 3.0
        and fp32_per_shape_max_cv < 3.0
        and quality_aggregates["fp32_bf16_fastmath"]["mean_embedding_cosine"]["minimum"] >= 0.99
    )

    qint8_summary = comparison_aggregates["qint8_bf16_vs_control"]
    qint8_material_gate = all(
        run["comparisons"]["qint8_bf16_vs_control"]["geometric_mean_latency_speedup"] >= 1.03
        for run in run_summaries
    )
    qint8_shape_follow_up = (
        any(item["comparisons"]["qint8_bf16_vs_control"]["median"] >= 1.03 for item in per_shape)
        and qint8_summary["minimum_speedup_across_all_runs_and_shapes"] >= 0.97
    )

    revisions = sorted({run["code_revision"] for run in run_summaries})
    result = {
        "schema_version": "armbench-bf16-aggregate/v1",
        "experiment_id": BF16_EXPERIMENT_ID,
        "analysis": {
            "code_revision": analysis_code_revision
            or os.getenv("GITHUB_SHA")
            or "working-tree",
        },
        "run_count": len(results),
        "github_run_ids": [run["github_run_id"] for run in run_summaries],
        "code_revisions": revisions,
        "contract": _contract_signature(first),
        "run_summaries": run_summaries,
        "comparisons": comparison_aggregates,
        "per_shape": per_shape,
        "quality": quality_aggregates,
        "stability": {
            "fp32_maximum_per_shape_speedup_cv_percent": fp32_per_shape_max_cv,
            "required_independent_runs": 5,
            "maximum_allowed_run_cv_percent": 3.0,
        },
        "decision": {
            "overall_status": (
                "performance-repetition-gate-passed-needs-task-quality"
                if fp32_performance_gate
                else "performance-repetition-gate-not-passed"
            ),
            "promotion_ready": False,
            "fp32_bf16": {
                "performance_gate_passed": fp32_performance_gate,
                "status": (
                    "retain-for-pinned-sts-and-retrieval-gate"
                    if fp32_performance_gate
                    else "reject-or-repeat"
                ),
            },
            "qint8_bf16": {
                "material_aggregate_effect_gate_passed": qint8_material_gate,
                "status": (
                    "retain-as-universal-candidate"
                    if qint8_material_gate
                    else (
                        "shape-specific-follow-up"
                        if qint8_shape_follow_up
                        else "reject-no-material-effect"
                    )
                ),
            },
            "remaining_gate": (
                "Run the pinned STS and retrieval evaluation before changing a default or "
                "headline. Review the implementation equivalence of every listed code revision."
            ),
        },
    }
    return result


def analyze_bf16_evidence(
    evidence_paths: Sequence[Path],
    *,
    analysis_code_revision: str | None = None,
) -> dict[str, Any]:
    """Load and aggregate BF16 experiment files."""

    evidence = [load_bf16_experiment(path) for path in evidence_paths]
    return analyze_bf16_results(
        evidence,
        analysis_code_revision=analysis_code_revision,
    )


def render_bf16_aggregate_markdown(result: Mapping[str, Any]) -> str:
    """Render the repeated-run decision and shape stability."""

    lines = [
        "# BF16 repeated-run analysis",
        "",
        f"- Experiment: `{result['experiment_id']}`",
        f"- Analysis code: `{result['analysis']['code_revision']}`",
        f"- Independent native runs: **{result['run_count']}**",
        f"- Decision: **{result['decision']['overall_status']}**",
        "",
        "## Run-level result",
        "",
        "| Run | Revision | FP32 + BF16 GM | FP32 minimum | QInt8 + BF16 GM | QInt8 minimum |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for run in result["run_summaries"]:
        fp32 = run["comparisons"]["fp32_bf16_vs_control"]
        qint8 = run["comparisons"]["qint8_bf16_vs_control"]
        lines.append(
            "| [{run}]({url}) | `{revision}` | {fp32:.4f}x | {fp32_min:.4f}x | "
            "{qint8:.4f}x | {qint8_min:.4f}x |".format(
                run=run["github_run_id"],
                url=run["github_run_url"],
                revision=str(run["code_revision"])[:7],
                fp32=fp32["geometric_mean_latency_speedup"],
                fp32_min=run["minimum_shape_speedups"]["fp32_bf16_vs_control"],
                qint8=qint8["geometric_mean_latency_speedup"],
                qint8_min=run["minimum_shape_speedups"]["qint8_bf16_vs_control"],
            )
        )

    lines.extend(
        [
            "",
            "## Shape stability",
            "",
            "| Batch | Seq | FP32 + BF16 median | CV | QInt8 + BF16 median | CV |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in result["per_shape"]:
        fp32 = case["comparisons"]["fp32_bf16_vs_control"]
        qint8 = case["comparisons"]["qint8_bf16_vs_control"]
        lines.append(
            f"| {case['batch_size']} | {case['sequence_length']} | {fp32['median']:.4f}x "
            f"| {fp32['cv_percent']:.2f}% | {qint8['median']:.4f}x "
            f"| {qint8['cv_percent']:.2f}% |"
        )

    fp32_aggregate = result["comparisons"]["fp32_bf16_vs_control"]
    qint8_aggregate = result["comparisons"]["qint8_bf16_vs_control"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "FP32 + BF16 run-level geometric-mean speedup: median "
                f"**{fp32_aggregate['run_geometric_mean_speedup']['median']:.4f}x**, "
                f"CV **{fp32_aggregate['run_geometric_mean_speedup']['cv_percent']:.2f}%**."
            ),
            "",
            (
                "QInt8 + BF16 run-level geometric-mean speedup: median "
                f"**{qint8_aggregate['run_geometric_mean_speedup']['median']:.4f}x**, "
                f"CV **{qint8_aggregate['run_geometric_mean_speedup']['cv_percent']:.2f}%**."
            ),
            "",
            f"- FP32 + BF16: **{result['decision']['fp32_bf16']['status']}**",
            f"- QInt8 + BF16: **{result['decision']['qint8_bf16']['status']}**",
            "- Default/headline promotion: **blocked on pinned STS and retrieval quality**",
            "",
        ]
    )
    return "\n".join(lines)


def write_bf16_aggregate(result: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write aggregate JSON and Markdown evidence."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "aggregate.json",
        "markdown": output_dir / "aggregate.md",
    }
    paths["json"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    paths["markdown"].write_text(
        render_bf16_aggregate_markdown(result),
        encoding="utf-8",
    )
    return paths
