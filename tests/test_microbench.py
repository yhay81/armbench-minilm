from armbench_minilm.microbench import _ranges, measure_memory_bandwidth


def test_ranges_cover_size_without_overlap() -> None:
    assert _ranges(10, 3) == [(0, 4), (4, 7), (7, 10)]


def test_memory_microbenchmark_retains_samples() -> None:
    result = measure_memory_bandwidth(
        sizes_mib=[1],
        thread_counts=[1, 2],
        warmups=0,
        iterations=2,
    )

    assert len(result["results"]) == 2
    assert all(len(item["samples_gbps"]) == 2 for item in result["results"])
    assert all(item["median_gbps"] > 0.0 for item in result["results"])
