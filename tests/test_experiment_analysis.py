from __future__ import annotations

import json
from pathlib import Path

import pytest

from armbench_minilm.constants import BF16_EXPERIMENT_ID
from armbench_minilm.experiment_analysis import (
    analyze_bf16_results,
    render_bf16_aggregate_markdown,
)


def _result(run_id: int, *, fp32_speedup: float = 1.32, threads: int = 4) -> dict:
    comparison = lambda speedup: {  # noqa: E731 - compact fixture factory
        "geometric_mean_latency_speedup": speedup,
        "minimum_case_speedup": speedup,
        "maximum_case_speedup": speedup,
    }
    quality = {
        "mean_embedding_cosine": 0.999,
        "minimum_embedding_cosine": 0.998,
        "mean_pairwise_similarity_error": 0.001,
        "maximum_pairwise_similarity_error": 0.002,
    }
    case_comparison = lambda speedup: {  # noqa: E731 - compact fixture factory
        "median_latency_speedup": speedup,
        "speedup_ci95_low": speedup - 0.01,
        "speedup_ci95_high": speedup + 0.01,
    }
    return {
        "schema_version": "armbench-experiment/v1",
        "experiment": {
            "id": BF16_EXPERIMENT_ID,
            "parent": "parent",
            "code_revision": "a" * 40,
        },
        "source_model": {"id": "model", "revision": "revision", "license": "Apache-2.0"},
        "configuration": {
            "batch_sizes": [1],
            "sequence_lengths": [16],
            "intra_op_threads": threads,
        },
        "machine": {
            "architecture": "aarch64",
            "onnxruntime": "1.28.0",
            "github_run_id": str(run_id),
            "linux_cpu": {"identity": {"cpu_implementer": "0x41", "cpu_part": "0xd49"}},
        },
        "variants": {
            "fp32_control": {
                "model_kind": "fp32",
                "session_config": {},
                "sha256": "1" * 64,
            },
            "fp32_bf16_fastmath": {
                "model_kind": "fp32",
                "session_config": {"bf16": "1"},
                "sha256": "1" * 64,
            },
            "qint8_control": {
                "model_kind": "qint8",
                "session_config": {},
                "sha256": "2" * 64,
            },
            "qint8_bf16_fastmath": {
                "model_kind": "qint8",
                "session_config": {"bf16": "1"},
                "sha256": "2" * 64,
            },
        },
        "quality_by_variant": {
            "fp32_bf16_fastmath": quality,
            "qint8_bf16_fastmath": quality,
        },
        "summary": {
            "comparisons": {
                "fp32_bf16_vs_control": comparison(fp32_speedup),
                "qint8_bf16_vs_control": comparison(1.02),
            }
        },
        "cases": [
            {
                "batch_size": 1,
                "sequence_length": 16,
                "comparisons": {
                    "fp32_bf16_vs_control": case_comparison(fp32_speedup),
                    "qint8_bf16_vs_control": case_comparison(1.04),
                },
            }
        ],
        "verdict": {"status": "needs-independent-native-repeats"},
    }


def _evidence(tmp_path: Path, results: list[dict]) -> list[tuple[Path, dict]]:
    evidence = []
    for index, result in enumerate(results):
        path = tmp_path / f"run-{index}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        evidence.append((path, result))
    return evidence


def test_five_stable_runs_pass_fp32_performance_gate(tmp_path: Path) -> None:
    results = [_result(index, fp32_speedup=1.32 + index * 0.001) for index in range(5)]

    aggregate = analyze_bf16_results(
        _evidence(tmp_path, results),
        analysis_code_revision="analysis-revision",
    )

    assert aggregate["run_count"] == 5
    assert aggregate["analysis"]["code_revision"] == "analysis-revision"
    assert aggregate["decision"]["overall_status"] == (
        "performance-repetition-gate-passed-needs-task-quality"
    )
    assert aggregate["decision"]["fp32_bf16"]["performance_gate_passed"] is True
    assert aggregate["decision"]["qint8_bf16"]["status"] == "shape-specific-follow-up"
    assert aggregate["decision"]["promotion_ready"] is False
    assert "blocked on pinned STS" in render_bf16_aggregate_markdown(aggregate)


def test_contract_mismatch_is_rejected(tmp_path: Path) -> None:
    results = [_result(1), _result(2, threads=2)]

    with pytest.raises(ValueError, match="contract mismatch"):
        analyze_bf16_results(_evidence(tmp_path, results))
