"""Command-line entry point for model preparation and benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from armbench_minilm.bounds import analyze_size_bounds
from armbench_minilm.models import existing_models, prepare_models


def _add_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(".armbench"),
        help="Model/cache directory (default: .armbench)",
    )


def _add_benchmark_options(parser: argparse.ArgumentParser) -> None:
    _add_paths(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--block-warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--measurement-blocks", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=20260811)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-inferences", type=int, default=20)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="armbench-minilm",
        description="Prepare and benchmark a pinned FP32/INT8 MiniLM pair.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Download FP32 and create INT8 ONNX models")
    _add_paths(prepare)
    prepare.add_argument("--force", action="store_true")

    benchmark = subparsers.add_parser("benchmark", help="Benchmark already prepared models")
    _add_benchmark_options(benchmark)

    bounds = subparsers.add_parser("bounds", help="Calculate model-size precision bounds")
    _add_paths(bounds)
    bounds.add_argument("--output", type=Path)

    ceiling = subparsers.add_parser(
        "ceiling", help="Calculate operator-scope Amdahl and preliminary roofline bounds"
    )
    _add_paths(ceiling)
    ceiling.add_argument("--evidence-dirs", type=Path, nargs="+", required=True)
    ceiling.add_argument("--memory-microbenchmark", type=Path, nargs="+")
    ceiling.add_argument("--output-dir", type=Path, default=Path("results/ceiling"))

    microbench = subparsers.add_parser(
        "microbench", help="Measure platform memory-copy bandwidth"
    )
    microbench.add_argument("--output", type=Path, default=Path("results/memory-bandwidth.json"))
    microbench.add_argument("--sizes-mib", type=int, nargs="+", default=[32, 256])
    microbench.add_argument("--thread-counts", type=int, nargs="+", default=[1, 4])
    microbench.add_argument("--warmups", type=int, default=3)
    microbench.add_argument("--iterations", type=int, default=10)

    all_command = subparsers.add_parser("all", help="Prepare, benchmark, and write reports")
    _add_benchmark_options(all_command)
    all_command.add_argument("--force", action="store_true")
    return parser


def _benchmark(args: argparse.Namespace, *, prepare: bool) -> None:
    from armbench_minilm.benchmark import run_benchmark
    from armbench_minilm.report import write_reports

    paths = (
        prepare_models(args.work_dir, force=args.force)
        if prepare
        else existing_models(args.work_dir)
    )
    result = run_benchmark(
        paths,
        batch_sizes=args.batch_sizes,
        sequence_lengths=args.sequence_lengths,
        warmups=args.warmups,
        block_warmups=args.block_warmups,
        iterations=args.iterations,
        threads=args.threads,
        measurement_blocks=args.measurement_blocks,
        random_seed=args.random_seed,
        bootstrap_resamples=args.bootstrap_resamples,
        profile_dir=args.output_dir / "profiles" if args.profile else None,
        profile_inferences=args.profile_inferences,
    )
    reports = write_reports(result, args.output_dir)
    print(f"Benchmark complete: {reports['markdown']}")


def _ceiling(args: argparse.Namespace) -> None:
    from armbench_minilm.ceiling import analyze_ceiling_runs, write_ceiling_reports

    paths = existing_models(args.work_dir)
    result = analyze_ceiling_runs(
        paths.baseline,
        args.evidence_dirs,
        memory_microbenchmark_paths=args.memory_microbenchmark,
    )
    reports = write_ceiling_reports(result, args.output_dir)
    print(f"Ceiling analysis complete: {reports['markdown']}")


def _microbench(args: argparse.Namespace) -> None:
    from armbench_minilm.microbench import measure_memory_bandwidth, write_memory_microbenchmark

    result = measure_memory_bandwidth(
        sizes_mib=args.sizes_mib,
        thread_counts=args.thread_counts,
        warmups=args.warmups,
        iterations=args.iterations,
    )
    output = write_memory_microbenchmark(result, args.output)
    print(f"Memory microbenchmark complete: {output.resolve()}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        paths = prepare_models(args.work_dir, force=args.force)
        print(f"Baseline:  {paths.baseline}")
        print(f"Optimized: {paths.optimized}")
    elif args.command == "benchmark":
        _benchmark(args, prepare=False)
    elif args.command == "bounds":
        paths = existing_models(args.work_dir)
        result = analyze_size_bounds(paths.baseline, candidate_path=paths.optimized)
        rendered = json.dumps(result, indent=2) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    elif args.command == "ceiling":
        _ceiling(args)
    elif args.command == "microbench":
        _microbench(args)
    else:
        _benchmark(args, prepare=True)


if __name__ == "__main__":
    main()
