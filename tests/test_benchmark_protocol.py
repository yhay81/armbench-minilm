from typing import Any

import numpy as np

from armbench_minilm import benchmark


def test_trial_sizes_preserve_total_iterations() -> None:
    assert benchmark._trial_sizes(11, 3) == [4, 4, 3]


def test_randomized_blocks_preserve_order_and_raw_samples(monkeypatch) -> None:
    monkeypatch.setattr(
        benchmark,
        "_embed",
        lambda session, feeds: np.zeros((1, 2), dtype=np.float32),
    )
    sessions: dict[str, Any] = {"baseline": object(), "optimized": object()}
    feeds: dict[str, Any] = {
        "baseline": {"input_ids": np.zeros((1, 4), dtype=np.int64)},
        "optimized": {"input_ids": np.zeros((1, 4), dtype=np.int64)},
    }

    samples, blocks = benchmark._measure_randomized_blocks(
        sessions,
        feeds,
        warmups=2,
        iterations=11,
        measurement_blocks=3,
        seed=123,
    )

    assert len(samples["baseline"]) == 11
    assert len(samples["optimized"]) == 11
    assert [block["iterations_per_model"] for block in blocks] == [4, 4, 3]
    assert all(set(block["order"]) == {"baseline", "optimized"} for block in blocks)
    assert all(
        len(block["samples_ms"][name]) == block["iterations_per_model"]
        for block in blocks
        for name in ("baseline", "optimized")
    )
