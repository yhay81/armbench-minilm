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


def summarize_latencies(latencies_ms: Sequence[float], *, batch_size: int) -> dict[str, float]:
    """Summarize measured latencies and derive sentence throughput."""

    if not latencies_ms:
        raise ValueError("at least one latency measurement is required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    values = np.asarray(latencies_ms, dtype=np.float64)
    median_ms = float(np.median(values))
    return {
        "median_ms": median_ms,
        "p95_ms": float(np.percentile(values, 95)),
        "mean_ms": float(np.mean(values)),
        "stdev_ms": float(np.std(values)),
        "sentences_per_second": float(batch_size / (median_ms / 1000.0)),
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
