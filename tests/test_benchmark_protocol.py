from typing import Any

import numpy as np

from armbench_minilm import benchmark


def test_trial_sizes_preserve_total_iterations() -> None:
    assert benchmark._trial_sizes(11, 3) == [4, 4, 3]


def test_process_cpu_summary_handles_coarse_zero_resolution() -> None:
    summary = benchmark._summarize_process_cpu(
        [0.0] * 5,
        bootstrap_seed=1,
        bootstrap_resamples=10,
    )

    assert summary["median_ms"] == 0.0
    assert summary["zero_sample_count"] == 5
    assert summary["median_ci95_low_ms"] is None


def test_randomized_blocks_preserve_order_and_raw_samples(monkeypatch) -> None:
    embed_calls = 0

    def fake_embed(session: Any, feeds: Any) -> np.ndarray[Any, Any]:
        nonlocal embed_calls
        embed_calls += 1
        return np.zeros((1, 2), dtype=np.float32)

    monkeypatch.setattr(
        benchmark,
        "_embed",
        fake_embed,
    )
    sessions: dict[str, Any] = {"baseline": object(), "optimized": object()}
    feeds: dict[str, Any] = {
        "baseline": {"input_ids": np.zeros((1, 4), dtype=np.int64)},
        "optimized": {"input_ids": np.zeros((1, 4), dtype=np.int64)},
    }

    samples, process_cpu_samples, blocks = benchmark._measure_randomized_blocks(
        sessions,
        feeds,
        warmups=2,
        block_warmups=2,
        iterations=11,
        measurement_blocks=3,
        seed=123,
    )

    assert len(samples["baseline"]) == 11
    assert len(samples["optimized"]) == 11
    assert len(process_cpu_samples["baseline"]) == 11
    assert len(process_cpu_samples["optimized"]) == 11
    assert embed_calls == 2 * 2 + 3 * 2 * 2 + 11 * 2
    assert [block["iterations_per_model"] for block in blocks] == [4, 4, 3]
    assert all(set(block["order"]) == {"baseline", "optimized"} for block in blocks)
    assert all(block["discarded_warmups_per_model"] == 2 for block in blocks)
    first_counts = {
        name: sum(block["order"][0] == name for block in blocks)
        for name in ("baseline", "optimized")
    }
    assert abs(first_counts["baseline"] - first_counts["optimized"]) == 1
    assert all(
        len(block["samples_ms"][name]) == block["iterations_per_model"]
        for block in blocks
        for name in ("baseline", "optimized")
    )
    assert all(
        len(block["process_cpu_samples_ms"][name]) == block["iterations_per_model"]
        for block in blocks
        for name in ("baseline", "optimized")
    )
