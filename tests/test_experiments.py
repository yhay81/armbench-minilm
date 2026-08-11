from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from armbench_minilm import benchmark
from armbench_minilm.experiments import (
    BF16_FASTMATH_KEY,
    bf16_variants,
    evaluate_bf16_verdict,
)
from armbench_minilm.models import ModelPaths


def test_bf16_variants_change_only_the_session_flag() -> None:
    paths = ModelPaths(Path("fp32.onnx"), Path("qint8.onnx"))

    variants = {variant.name: variant for variant in bf16_variants(paths)}

    assert set(variants) == {
        "fp32_control",
        "fp32_bf16_fastmath",
        "qint8_control",
        "qint8_bf16_fastmath",
    }
    assert variants["fp32_control"].model_path == variants["fp32_bf16_fastmath"].model_path
    assert variants["qint8_control"].model_path == variants["qint8_bf16_fastmath"].model_path
    assert variants["fp32_control"].session_config == {}
    assert variants["qint8_control"].session_config == {}
    assert variants["fp32_bf16_fastmath"].session_config == {BF16_FASTMATH_KEY: "1"}
    assert variants["qint8_bf16_fastmath"].session_config == {BF16_FASTMATH_KEY: "1"}


def test_create_session_records_experiment_session_config(monkeypatch, tmp_path: Path) -> None:
    created_options: list[Any] = []

    class FakeSessionOptions:
        def __init__(self) -> None:
            self.entries: dict[str, str] = {}
            created_options.append(self)

        def add_session_config_entry(self, key: str, value: str) -> None:
            self.entries[key] = value

    monkeypatch.setattr(benchmark.ort, "SessionOptions", FakeSessionOptions)
    monkeypatch.setattr(benchmark.ort, "InferenceSession", lambda *args, **kwargs: object())

    benchmark._create_session(
        tmp_path / "model.onnx",
        threads=4,
        session_config={BF16_FASTMATH_KEY: "1"},
    )

    assert created_options[0].entries[BF16_FASTMATH_KEY] == "1"
    assert created_options[0].entries["session.intra_op.spin_duration_us"] == "1000"


def _verdict_result(*, speedup: float, minimum: float, cosine: float = 0.999) -> dict:
    comparison = {
        "geometric_mean_latency_speedup": speedup,
        "minimum_case_speedup": minimum,
        "maximum_case_speedup": speedup,
    }
    return {
        "machine": {
            "architecture": "aarch64",
            "linux_cpu": {"features": ["bf16", "svebf16"]},
        },
        "quality_by_variant": {
            "fp32_bf16_fastmath": {"mean_embedding_cosine": cosine},
            "qint8_bf16_fastmath": {"mean_embedding_cosine": cosine},
        },
        "summary": {
            "comparisons": {
                "fp32_bf16_vs_control": comparison,
                "qint8_bf16_vs_control": comparison,
            }
        },
    }


def test_bf16_verdict_requires_native_repeats_after_single_run_success() -> None:
    verdict = evaluate_bf16_verdict(_verdict_result(speedup=1.05, minimum=0.98))

    assert verdict["status"] == "needs-independent-native-repeats"
    assert verdict["promotion_ready"] is False


@pytest.mark.parametrize(
    ("speedup", "minimum", "expected"),
    [
        (1.05, 0.95, "shape-specific-follow-up"),
        (1.005, 1.0, "rejected-no-material-effect"),
        (1.02, 1.0, "needs-follow-up"),
    ],
)
def test_bf16_verdict_applies_predeclared_effect_thresholds(
    speedup: float,
    minimum: float,
    expected: str,
) -> None:
    verdict = evaluate_bf16_verdict(_verdict_result(speedup=speedup, minimum=minimum))

    assert verdict["status"] == expected


def test_bf16_verdict_rejects_quality_failure() -> None:
    verdict = evaluate_bf16_verdict(_verdict_result(speedup=1.10, minimum=1.02, cosine=0.98))

    assert verdict["status"] == "rejected-quality-gate"
