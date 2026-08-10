"""Command-line entry point for model preparation and benchmarking."""

from __future__ import annotations

import argparse
from pathlib import Path

from armbench_minilm.benchmark import run_benchmark
from armbench_minilm.models import existing_models, prepare_models
from armbench_minilm.report import write_reports


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
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)


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

    all_command = subparsers.add_parser("all", help="Prepare, benchmark, and write reports")
    _add_benchmark_options(all_command)
    all_command.add_argument("--force", action="store_true")
    return parser


def _benchmark(args: argparse.Namespace, *, prepare: bool) -> None:
    paths = (
        prepare_models(args.work_dir, force=args.force)
        if prepare
        else existing_models(args.work_dir)
    )
    result = run_benchmark(
        paths,
        batch_sizes=args.batch_sizes,
        warmups=args.warmups,
        iterations=args.iterations,
        threads=args.threads,
    )
    reports = write_reports(result, args.output_dir)
    print(f"Benchmark complete: {reports['markdown']}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        paths = prepare_models(args.work_dir, force=args.force)
        print(f"Baseline:  {paths.baseline}")
        print(f"Optimized: {paths.optimized}")
    elif args.command == "benchmark":
        _benchmark(args, prepare=False)
    else:
        _benchmark(args, prepare=True)


if __name__ == "__main__":
    main()
