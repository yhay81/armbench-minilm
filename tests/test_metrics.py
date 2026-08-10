from __future__ import annotations

import numpy as np
import pytest

from armbench_minilm.metrics import (
    bootstrap_median_ci,
    bootstrap_speedup_ci,
    mean_pool,
    normalize_rows,
    quality_metrics,
    rowwise_cosine,
    summarize_latencies,
)


def test_mean_pool_ignores_padding() -> None:
    embeddings = np.asarray([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]], dtype=np.float32)
    mask = np.asarray([[1, 1, 0]], dtype=np.int64)

    assert np.allclose(mean_pool(embeddings, mask), [[2.0, 4.0]])


@pytest.mark.parametrize(
    ("embeddings", "mask"),
    [
        (np.zeros((2, 3)), np.zeros((2, 3))),
        (np.zeros((2, 3, 4)), np.zeros((2, 3, 1))),
        (np.zeros((2, 3, 4)), np.zeros((2, 4))),
    ],
)
def test_mean_pool_rejects_invalid_shapes(embeddings: np.ndarray, mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        mean_pool(embeddings, mask)


def test_normalize_rows_handles_zero_vector() -> None:
    values = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    assert np.allclose(normalize_rows(values), [[0.6, 0.8], [0.0, 0.0]])


def test_rowwise_cosine_requires_equal_shapes() -> None:
    with pytest.raises(ValueError):
        rowwise_cosine(np.zeros((2, 3)), np.zeros((3, 2)))


def test_latency_summary_uses_median_for_throughput() -> None:
    result = summarize_latencies([10.0, 20.0, 30.0], batch_size=2)

    assert result["median_ms"] == pytest.approx(20.0)
    assert result["sentences_per_second"] == pytest.approx(100.0)


def test_bootstrap_intervals_are_exact_for_constant_samples() -> None:
    median = bootstrap_median_ci([10.0] * 20, seed=7, resamples=100)
    speedup = bootstrap_speedup_ci([10.0] * 20, [5.0] * 20, seed=7, resamples=100)

    assert median["median_ci95_low_ms"] == pytest.approx(10.0)
    assert median["median_ci95_high_ms"] == pytest.approx(10.0)
    assert median["median_ci95_half_width_percent"] == pytest.approx(0.0)
    assert speedup["speedup_ci95_low"] == pytest.approx(2.0)
    assert speedup["speedup_ci95_high"] == pytest.approx(2.0)


def test_quality_metrics_are_ideal_for_identical_embeddings() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    result = quality_metrics(embeddings, embeddings.copy())

    assert result["mean_embedding_cosine"] == pytest.approx(1.0)
    assert result["minimum_embedding_cosine"] == pytest.approx(1.0)
    assert result["mean_pairwise_similarity_error"] == pytest.approx(0.0)
    assert result["maximum_pairwise_similarity_error"] == pytest.approx(0.0)
