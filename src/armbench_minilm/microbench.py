"""Small, reproducible platform microbenchmarks used by the roofline analysis."""

from __future__ import annotations

import json
import platform
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import psutil


def _ranges(size: int, parts: int) -> list[tuple[int, int]]:
    if size < 1 or parts < 1:
        raise ValueError("size and parts must be positive")
    base, remainder = divmod(size, parts)
    result: list[tuple[int, int]] = []
    start = 0
    for index in range(parts):
        end = start + base + (1 if index < remainder else 0)
        result.append((start, end))
        start = end
    return result


def _copy_parallel(
    destination: np.ndarray[Any, Any],
    source: np.ndarray[Any, Any],
    ranges: list[tuple[int, int]],
    executor: ThreadPoolExecutor | None,
) -> None:
    if executor is None:
        np.copyto(destination, source)
        return
    futures = [
        executor.submit(np.copyto, destination[start:end], source[start:end])
        for start, end in ranges
    ]
    for future in futures:
        future.result()


def measure_memory_bandwidth(
    *,
    sizes_mib: list[int],
    thread_counts: list[int],
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    """Measure read-plus-write copy bandwidth with retained raw samples."""

    if not sizes_mib or any(size < 1 for size in sizes_mib):
        raise ValueError("sizes_mib must contain positive integers")
    if not thread_counts or any(threads < 1 for threads in thread_counts):
        raise ValueError("thread_counts must contain positive integers")
    if warmups < 0 or iterations < 1:
        raise ValueError("warmups must be non-negative and iterations must be positive")

    results: list[dict[str, Any]] = []
    for size_mib in sizes_mib:
        size_bytes = size_mib * 2**20
        source = np.empty(size_bytes, dtype=np.uint8)
        destination = np.empty_like(source)
        source.fill(0xA5)
        destination.fill(0)
        for threads in thread_counts:
            ranges = _ranges(size_bytes, threads)
            executor = ThreadPoolExecutor(max_workers=threads) if threads > 1 else None
            try:
                for _ in range(warmups):
                    _copy_parallel(destination, source, ranges, executor)
                samples_gbps: list[float] = []
                samples_ms: list[float] = []
                effective_bytes = size_bytes * 2
                for _ in range(iterations):
                    started = perf_counter()
                    _copy_parallel(destination, source, ranges, executor)
                    elapsed = perf_counter() - started
                    samples_ms.append(elapsed * 1_000.0)
                    samples_gbps.append(effective_bytes / elapsed / 1_000_000_000.0)
            finally:
                if executor is not None:
                    executor.shutdown()
            results.append(
                {
                    "size_mib": size_mib,
                    "size_bytes": size_bytes,
                    "working_set_bytes": size_bytes * 2,
                    "threads": threads,
                    "effective_bytes_per_iteration": effective_bytes,
                    "median_gbps": statistics.median(samples_gbps),
                    "minimum_gbps": min(samples_gbps),
                    "maximum_gbps": max(samples_gbps),
                    "median_ms": statistics.median(samples_ms),
                    "samples_gbps": samples_gbps,
                    "samples_ms": samples_ms,
                }
            )
    return {
        "schema_version": 1,
        "benchmark": "NumPy copy read-plus-write bandwidth",
        "configuration": {
            "sizes_mib": sizes_mib,
            "thread_counts": thread_counts,
            "warmups": warmups,
            "iterations": iterations,
            "decimal_gigabytes": True,
        },
        "machine": {
            "architecture": platform.machine(),
            "operating_system": platform.platform(),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "memory_bytes": psutil.virtual_memory().total,
        },
        "results": results,
        "notes": [
            "Effective traffic counts one source read and one destination write.",
            "The smaller working set is cache-residency-oriented, not proof of a cache hit rate.",
            "The largest working set is used for the preliminary DRAM-bandwidth projection.",
        ],
    }


def write_memory_microbenchmark(result: dict[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output
