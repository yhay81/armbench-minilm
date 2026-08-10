"""Numerically testable metrics shared by the benchmark and reports."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def mean_pool(
    token_embeddings: NDArray[np.floating],
    attention_mask: NDArray[np.integer],
) -> NDArray[np.float32]:
    """Mean-pool token embeddings while excluding padded positions."""

    if token_embeddings.ndim != 3:
        raise ValueError("token_embeddings must have shape [batch, sequence, hidden]")
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if token_embeddings.shape[:2] != attention_mask.shape:
        raise ValueError("attention_mask shape must match the first two embedding dimensions")

    mask = attention_mask.astype(np.float32, copy=False)[..., None]
    summed = np.sum(token_embeddings.astype(np.float32, copy=False) * mask, axis=1)
    counts = np.clip(np.sum(mask, axis=1), 1e-9, None)
    return (summed / counts).astype(np.float32, copy=False)


def normalize_rows(values: NDArray[np.floating]) -> NDArray[np.float32]:
    """L2-normalize a two-dimensional matrix row by row."""

    if values.ndim != 2:
        raise ValueError("values must have shape [rows, dimensions]")
    floats = values.astype(np.float32, copy=False)
    norms = np.linalg.norm(floats, axis=1, keepdims=True)
    return (floats / np.clip(norms, 1e-12, None)).astype(np.float32, copy=False)


def rowwise_cosine(
    baseline: NDArray[np.floating],
    candidate: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Cosine similarity between corresponding rows of two matrices."""

    if baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate must have identical shapes")
    left = normalize_rows(baseline)
    right = normalize_rows(candidate)
    return np.sum(left * right, axis=1).astype(np.float32, copy=False)


def summarize_latencies(
    latencies_ms: Sequence[float],
    *,
    batch_size: int,
    bootstrap_seed: int = 0,
    bootstrap_resamples: int = 2_000,
) -> dict[str, float]:
    """Summarize measured latencies and derive sentence throughput."""

    if not latencies_ms:
        raise ValueError("at least one latency measurement is required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    values = np.asarray(latencies_ms, dtype=np.float64)
    median_ms = float(np.median(values))
    confidence = bootstrap_median_ci(
        latencies_ms,
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
    )
    return {
        "median_ms": median_ms,
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(np.mean(values)),
        "stdev_ms": float(np.std(values)),
        "sentences_per_second": float(batch_size / (median_ms / 1000.0)),
        **confidence,
    }


def bootstrap_median_ci(
    latencies_ms: Sequence[float],
    *,
    seed: int,
    resamples: int = 2_000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Return a deterministic percentile-bootstrap interval for the median."""

    if not latencies_ms:
        raise ValueError("at least one latency measurement is required")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    values = np.asarray(latencies_ms, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(resamples, values.size), replace=True)
    medians = np.median(sampled, axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(medians, [alpha, 1.0 - alpha])
    median = float(np.median(values))
    half_width_percent = (
        float((high - low) / 2.0 / median * 100.0) if median > 0.0 else float("inf")
    )
    return {
        "median_ci95_low_ms": float(low),
        "median_ci95_high_ms": float(high),
        "median_ci95_half_width_percent": half_width_percent,
    }


def bootstrap_speedup_ci(
    baseline_latencies_ms: Sequence[float],
    candidate_latencies_ms: Sequence[float],
    *,
    seed: int,
    resamples: int = 2_000,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Bootstrap an interval for the ratio of baseline and candidate medians."""

    if not baseline_latencies_ms or not candidate_latencies_ms:
        raise ValueError("both latency samples must be non-empty")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    baseline = np.asarray(baseline_latencies_ms, dtype=np.float64)
    candidate = np.asarray(candidate_latencies_ms, dtype=np.float64)
    rng = np.random.default_rng(seed)
    baseline_samples = rng.choice(baseline, size=(resamples, baseline.size), replace=True)
    candidate_samples = rng.choice(candidate, size=(resamples, candidate.size), replace=True)
    ratios = np.median(baseline_samples, axis=1) / np.median(candidate_samples, axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(ratios, [alpha, 1.0 - alpha])
    return {
        "speedup_ci95_low": float(low),
        "speedup_ci95_high": float(high),
    }


def quality_metrics(
    baseline: NDArray[np.floating],
    candidate: NDArray[np.floating],
) -> dict[str, float]:
    """Measure embedding and pairwise-similarity preservation."""

    baseline_normalized = normalize_rows(baseline)
    candidate_normalized = normalize_rows(candidate)
    cosines = rowwise_cosine(baseline_normalized, candidate_normalized)
    baseline_pairs = baseline_normalized @ baseline_normalized.T
    candidate_pairs = candidate_normalized @ candidate_normalized.T
    pairwise_error = np.abs(baseline_pairs - candidate_pairs)
    return {
        "mean_embedding_cosine": float(np.mean(cosines)),
        "minimum_embedding_cosine": float(np.min(cosines)),
        "mean_pairwise_similarity_error": float(np.mean(pairwise_error)),
        "maximum_pairwise_similarity_error": float(np.max(pairwise_error)),
    }
