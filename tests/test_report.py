from __future__ import annotations

import json

from armbench_minilm.report import render_markdown, write_reports


def _result() -> dict:
    metrics = {
        "median_ms": 10.0,
        "p95_ms": 11.0,
        "mean_ms": 10.1,
        "stdev_ms": 0.2,
        "sentences_per_second": 100.0,
    }
    return {
        "schema_version": 1,
        "generated_at_utc": "2026-08-11T00:00:00+00:00",
        "machine": {
            "architecture": "aarch64",
            "processor": "test-cpu",
            "operating_system": "Linux-test",
            "python": "3.12.0",
            "onnxruntime": "1.22.0",
            "physical_cpu_cores": 4,
            "logical_cpu_cores": 4,
        },
        "configuration": {
            "intra_op_threads": 4,
            "quality_sentence_count": 32,
        },
        "models": {
            "baseline": {"size_bytes": 100 * 1024 * 1024, "sha256": "a" * 64},
            "optimized": {"size_bytes": 25 * 1024 * 1024, "sha256": "b" * 64},
        },
        "quality": {
            "mean_embedding_cosine": 0.999,
            "minimum_embedding_cosine": 0.998,
            "mean_pairwise_similarity_error": 0.001,
            "maximum_pairwise_similarity_error": 0.002,
        },
        "batches": [
            {
                "batch_size": 1,
                "baseline": metrics,
                "optimized": {**metrics, "median_ms": 5.0, "sentences_per_second": 200.0},
                "median_latency_speedup": 2.0,
                "throughput_gain_percent": 100.0,
            }
        ],
        "summary": {
            "model_size_reduction_percent": 75.0,
            "geometric_mean_latency_speedup": 2.0,
        },
    }


def test_markdown_contains_core_evidence() -> None:
    report = render_markdown(_result())

    assert "Architecture | `aarch64`" in report
    assert "Measured size reduction: 75.0%" in report
    assert "2.00x" in report
    assert "0.99900000" in report


def test_write_reports_creates_parseable_json_and_html(tmp_path) -> None:
    paths = write_reports(_result(), tmp_path)

    assert json.loads(paths["json"].read_text(encoding="utf-8"))["schema_version"] == 1
    assert "<!doctype html>" in paths["html"].read_text(encoding="utf-8")
    assert paths["markdown"].is_file()


def test_markdown_renders_fixed_shape_confidence_and_profile_evidence() -> None:
    result = _result()
    result["schema_version"] = 2
    result["configuration"].update(
        {
            "sequence_lengths": [16],
            "measurement_blocks_per_case": 5,
            "measured_iterations_per_model_and_batch": 100,
            "bootstrap_resamples": 2_000,
        }
    )
    case = result["batches"][0]
    case["sequence_length"] = 16
    case["speedup_ci95_low"] = 1.9
    case["speedup_ci95_high"] = 2.1
    for metrics in (case["baseline"], case["optimized"]):
        metrics["median_ci95_low_ms"] = metrics["median_ms"] * 0.99
        metrics["median_ci95_high_ms"] = metrics["median_ms"] * 1.01
        metrics["median_ci95_half_width_percent"] = 1.0
    result["summary"].update(
        {
            "maximum_median_ci95_half_width_percent": 1.0,
            "tail_spike_cases": [],
        }
    )
    result["profiles"] = [
        {
            "batch_size": 1,
            "sequence_length": 16,
            "baseline": {
                "operators": [{"operator": "MatMul", "duration_us": 100.0, "calls": 4}]
            },
            "optimized": {
                "operators": [
                    {"operator": "DynamicQuantizeMatMul", "duration_us": 50.0, "calls": 4}
                ]
            },
        }
    ]

    report = render_markdown(result)

    assert "balanced-randomized A/B blocks" in report
    assert "| 1 | 16 |" in report
    assert "2.00x [1.90, 2.10]" in report
    assert "Operator profile" in report
    assert "DynamicQuantizeMatMul" in report
