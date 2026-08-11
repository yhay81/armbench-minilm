from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from armbench_minilm.quality import (
    average_ranks,
    ndcg_at_k,
    quality_comparison_decision,
    render_quality_markdown,
    spearman_correlation,
    verify_file_hash,
    write_quality_evaluation,
)


def test_average_ranks_matches_average_tie_semantics() -> None:
    ranks = average_ranks([30.0, 10.0, 20.0, 20.0])

    np.testing.assert_allclose(ranks, [4.0, 1.0, 2.5, 2.5])


def test_spearman_correlation_handles_ties() -> None:
    correlation = spearman_correlation([1.0, 2.0, 2.0, 4.0], [10.0, 20.0, 20.0, 40.0])

    assert correlation == pytest.approx(1.0)


def test_ndcg_excludes_identical_query_id() -> None:
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    corpus = np.asarray(
        [
            [1.0, 0.0],
            [0.9, math.sqrt(1.0 - 0.9**2)],
            [0.8, 0.6],
        ],
        dtype=np.float32,
    )

    score = ndcg_at_k(
        query,
        corpus,
        ["query"],
        ["query", "relevant", "other"],
        {"query": {"relevant": 1.0}},
        k=2,
        exclude_identical_ids=True,
    )

    assert score["ndcg_at_2"] == pytest.approx(1.0)
    assert score["queries_with_relevant_in_top_2"] == 1


def test_ndcg_applies_log_discount() -> None:
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    corpus = np.asarray([[1.0, 0.0], [0.8, 0.6]], dtype=np.float32)

    score = ndcg_at_k(
        query,
        corpus,
        ["q"],
        ["distractor", "relevant"],
        {"q": {"relevant": 1.0}},
        k=2,
        exclude_identical_ids=False,
    )

    assert score["ndcg_at_2"] == pytest.approx(1.0 / math.log2(3.0))


def test_quality_comparison_applies_all_predeclared_gates() -> None:
    passing = quality_comparison_decision(
        control_sts_x100=70.0,
        candidate_sts_x100=69.6,
        control_retrieval=0.5,
        candidate_retrieval=0.495,
        sts_mean_cosine=0.999,
        retrieval_mean_cosine=0.998,
    )
    failing = quality_comparison_decision(
        control_sts_x100=70.0,
        candidate_sts_x100=69.49,
        control_retrieval=0.5,
        candidate_retrieval=0.494,
        sts_mean_cosine=0.989,
        retrieval_mean_cosine=0.998,
    )

    assert passing["passed"] is True
    assert failing["passed"] is False
    assert failing["gates"]["sts_absolute_loss_points_at_most_0_5"] is False
    assert failing["gates"]["retrieval_relative_loss_at_most_0_01"] is False
    assert failing["gates"]["sts_mean_embedding_cosine_at_least_0_99"] is False


def test_verify_file_hash_rejects_source_drift(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    path.write_text("{}\n", encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="source hash mismatch"):
        verify_file_hash(path, "0" * 64)


def _report_fixture() -> dict:
    variants = (
        "fp32_control",
        "fp32_bf16_fastmath",
        "qint8_control",
        "qint8_bf16_fastmath",
    )
    sts_scores = {
        name: {
            "macro_cosine_spearman_x100": 70.0,
            "by_config": {"en-hi": {"cosine_spearman_x100": 70.0}},
        }
        for name in variants
    }
    retrieval_scores = {name: {"ndcg_at_10_x100": 50.0} for name in variants}
    comparison = {
        "sts_absolute_loss_points": 0.0,
        "retrieval_relative_loss": 0.0,
        "sts_mean_corresponding_embedding_cosine": 1.0,
        "retrieval_mean_corresponding_embedding_cosine": 1.0,
        "status": "passed",
    }
    return {
        "experiment": {
            "id": "r2-bf16-task-quality-v1",
            "parent": "r2-bf16-fastmath-v1",
            "code_revision": "abc123",
        },
        "generated_at_utc": "2026-08-11T00:00:00+00:00",
        "machine": {"architecture": "aarch64"},
        "source_model": {"revision": "model-revision"},
        "tasks": {
            "indic_crosslingual_sts": {
                "pairs_per_configuration": 256,
                "scores_by_variant": sts_scores,
            },
            "arguana": {
                "corpus_rows": 8674,
                "query_rows": 1406,
                "unretrievable_qrel_target_count": 5,
                "scores_by_variant": retrieval_scores,
            },
        },
        "comparisons": {"fp32_bf16_vs_control": comparison},
        "verdict": {"status": "passed", "reason": "fixture"},
    }


def test_quality_report_writers_are_lf_stable(tmp_path: Path) -> None:
    result = _report_fixture()

    markdown = render_quality_markdown(result)
    paths = write_quality_evaluation(result, tmp_path)

    assert "Task scores" in markdown
    assert b"\r\n" not in paths["markdown"].read_bytes()
    assert b"\r\n" not in paths["json"].read_bytes()
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["verdict"]["status"] == (
        "passed"
    )
