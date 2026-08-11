"""Evidence-gated runtime experiments that do not replace the submitted baseline."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

from armbench_minilm.benchmark import (
    _batch,
    _create_session,
    _embed,
    _feeds,
    _machine_metadata,
    _measure_randomized_blocks,
    _model_metadata,
    _profile_model_case,
    _summarize_process_cpu,
    _tail_spike_cases,
    _trial_sizes,
)
from armbench_minilm.constants import (
    BENCHMARK_SENTENCES,
    BF16_EXPERIMENT_ID,
    BF16_FASTMATH_KEY,
    BF16_PARENT_EXPERIMENT,
    MAX_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
)
from armbench_minilm.metrics import (
    bootstrap_speedup_ci,
    quality_metrics,
    summarize_latencies,
)
from armbench_minilm.models import ModelPaths


@dataclass(frozen=True)
class SessionVariant:
    """One model artifact and session configuration in a controlled experiment."""

    name: str
    model_kind: str
    model_path: Path
    session_config: Mapping[str, str]


def bf16_variants(paths: ModelPaths) -> tuple[SessionVariant, ...]:
    """Return the four variants needed to isolate the Arm64 BF16 session flag."""

    return (
        SessionVariant("fp32_control", "fp32", paths.baseline, {}),
        SessionVariant(
            "fp32_bf16_fastmath",
            "fp32",
            paths.baseline,
            {BF16_FASTMATH_KEY: "1"},
        ),
        SessionVariant("qint8_control", "dynamic_qint8", paths.optimized, {}),
        SessionVariant(
            "qint8_bf16_fastmath",
            "dynamic_qint8",
            paths.optimized,
            {BF16_FASTMATH_KEY: "1"},
        ),
    )


def _comparison(
    samples: Mapping[str, Sequence[float]],
    *,
    control: str,
    candidate: str,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    control_values = samples[control]
    candidate_values = samples[candidate]
    speedup = float(np.median(control_values) / np.median(candidate_values))
    return {
        "control": control,
        "candidate": candidate,
        "median_latency_speedup": speedup,
        **bootstrap_speedup_ci(
            control_values,
            candidate_values,
            seed=seed,
            resamples=bootstrap_resamples,
        ),
    }


def _geometric_mean(values: Sequence[float]) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=np.float64)))))


def evaluate_bf16_verdict(result: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the predeclared experiment contract without promoting a single run."""

    architecture = str(result["machine"]["architecture"]).lower()
    features = set((result["machine"].get("linux_cpu") or {}).get("features", []))
    if architecture not in {"aarch64", "arm64"}:
        return {
            "status": "needs-native-arm64-evaluation",
            "promotion_ready": False,
            "reason": "BF16 fast-math must be evaluated on the native Arm64 target.",
        }
    if features and not features.intersection({"bf16", "svebf16"}):
        return {
            "status": "aborted-unsupported-hardware",
            "promotion_ready": False,
            "reason": "The recorded Arm64 CPU features do not advertise BF16 support.",
        }

    qualities = result["quality_by_variant"]
    bf16_quality_ok = all(
        qualities[name]["mean_embedding_cosine"] >= 0.99
        for name in ("fp32_bf16_fastmath", "qint8_bf16_fastmath")
    )
    if not bf16_quality_ok:
        return {
            "status": "rejected-quality-gate",
            "promotion_ready": False,
            "reason": "At least one BF16 candidate fell below mean embedding cosine 0.99.",
        }

    summaries = result["summary"]["comparisons"]
    same_artifact = {
        name: summaries[name] for name in ("fp32_bf16_vs_control", "qint8_bf16_vs_control")
    }
    best_name, best = max(
        same_artifact.items(),
        key=lambda item: item[1]["geometric_mean_latency_speedup"],
    )
    best_speedup = best["geometric_mean_latency_speedup"]
    if best_speedup >= 1.03 and best["minimum_case_speedup"] >= 0.97:
        return {
            "status": "needs-independent-native-repeats",
            "promotion_ready": False,
            "leading_comparison": best_name,
            "reason": (
                "The candidate passed the single-run effect and regression gates; "
                "five independent native runs are still required for promotion."
            ),
        }
    if best_speedup >= 1.03:
        return {
            "status": "shape-specific-follow-up",
            "promotion_ready": False,
            "leading_comparison": best_name,
            "reason": "The aggregate gain passed, but at least one shape regressed by over 3%.",
        }
    if all(item["geometric_mean_latency_speedup"] <= 1.01 for item in same_artifact.values()):
        return {
            "status": "rejected-no-material-effect",
            "promotion_ready": False,
            "reason": "Neither BF16 candidate improved its same-artifact control by over 1%.",
        }
    return {
        "status": "needs-follow-up",
        "promotion_ready": False,
        "leading_comparison": best_name,
        "reason": "The measured effect is between the rejection and success thresholds.",
    }


def run_bf16_experiment(
    paths: ModelPaths,
    *,
    batch_sizes: Sequence[int],
    sequence_lengths: Sequence[int],
    warmups: int,
    block_warmups: int,
    iterations: int,
    measurement_blocks: int,
    random_seed: int,
    bootstrap_resamples: int,
    threads: int,
    code_revision: str | None = None,
    profile_dir: Path | None = None,
    profile_inferences: int = 20,
) -> dict[str, Any]:
    """Compare FP32 and dynamic-QInt8 with and without Arm64 BF16 fast-math."""

    if not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers")
    if not sequence_lengths or any(
        length < 1 or length > MAX_LENGTH for length in sequence_lengths
    ):
        raise ValueError(f"sequence_lengths must be between 1 and {MAX_LENGTH}")
    if warmups < 0 or block_warmups < 0 or iterations < 1 or threads < 1:
        raise ValueError(
            "warmups and block_warmups must be non-negative; "
            "iterations and threads must be positive"
        )
    _trial_sizes(iterations, measurement_blocks)
    if bootstrap_resamples < 1 or profile_inferences < 1:
        raise ValueError("bootstrap_resamples and profile_inferences must be positive")

    variants = bf16_variants(paths)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=True,
    )
    sessions: dict[str, Any] = {}
    variant_metadata: dict[str, Any] = {}
    for variant in variants:
        session, load_ms = _create_session(
            variant.model_path,
            threads=threads,
            session_config=variant.session_config,
        )
        sessions[variant.name] = session
        variant_metadata[variant.name] = {
            "model_kind": variant.model_kind,
            "session_config": dict(variant.session_config),
            **_model_metadata(variant.model_path, load_ms=load_ms),
        }

    quality_embeddings: dict[str, Any] = {}
    for variant in variants:
        quality_feeds = _feeds(tokenizer, sessions[variant.name], BENCHMARK_SENTENCES)
        quality_embeddings[variant.name] = _embed(sessions[variant.name], quality_feeds)
    reference_embeddings = quality_embeddings["fp32_control"]
    quality_by_variant = {
        name: quality_metrics(reference_embeddings, embeddings)
        for name, embeddings in quality_embeddings.items()
    }

    comparison_specs = (
        ("fp32_bf16_vs_control", "fp32_control", "fp32_bf16_fastmath"),
        ("qint8_bf16_vs_control", "qint8_control", "qint8_bf16_fastmath"),
        ("qint8_control_vs_fp32", "fp32_control", "qint8_control"),
        ("qint8_bf16_vs_fp32", "fp32_control", "qint8_bf16_fastmath"),
    )
    cases: list[dict[str, Any]] = []
    case_index = 0
    for batch_size in batch_sizes:
        for sequence_length in sequence_lengths:
            sentences = _batch(BENCHMARK_SENTENCES, batch_size)
            feeds_by_variant = {
                variant.name: _feeds(
                    tokenizer,
                    sessions[variant.name],
                    sentences,
                    fixed_sequence_length=sequence_length,
                )
                for variant in variants
            }
            case_seed = random_seed + case_index * 1_009
            samples, process_cpu_samples, blocks = _measure_randomized_blocks(
                sessions,
                feeds_by_variant,
                warmups=warmups,
                block_warmups=block_warmups,
                iterations=iterations,
                measurement_blocks=measurement_blocks,
                seed=case_seed,
            )
            variant_metrics: dict[str, Any] = {}
            for variant_index, variant in enumerate(variants):
                variant_metrics[variant.name] = {
                    **summarize_latencies(
                        samples[variant.name],
                        batch_size=batch_size,
                        bootstrap_seed=case_seed + 10 + variant_index,
                        bootstrap_resamples=bootstrap_resamples,
                    ),
                    "process_cpu": _summarize_process_cpu(
                        process_cpu_samples[variant.name],
                        bootstrap_seed=case_seed + 20 + variant_index,
                        bootstrap_resamples=bootstrap_resamples,
                    ),
                }
            comparisons = {
                comparison_name: _comparison(
                    samples,
                    control=control,
                    candidate=candidate,
                    seed=case_seed + 30 + comparison_index,
                    bootstrap_resamples=bootstrap_resamples,
                )
                for comparison_index, (comparison_name, control, candidate) in enumerate(
                    comparison_specs
                )
            }
            cases.append(
                {
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "random_seed": case_seed,
                    "variants": variant_metrics,
                    "comparisons": comparisons,
                    "measurement_blocks": blocks,
                }
            )
            case_index += 1

    profiles: list[dict[str, Any]] = []
    if profile_dir is not None:
        profile_dir = profile_dir.resolve()
        for batch_size in batch_sizes:
            for sequence_length in sequence_lengths:
                sentences = _batch(BENCHMARK_SENTENCES, batch_size)
                profiles.append(
                    {
                        "batch_size": batch_size,
                        "sequence_length": sequence_length,
                        "variants": {
                            variant.name: _profile_model_case(
                                variant.model_path,
                                tokenizer,
                                sentences,
                                model_name=variant.name,
                                batch_size=batch_size,
                                sequence_length=sequence_length,
                                threads=threads,
                                profile_dir=profile_dir,
                                profiled_inferences=profile_inferences,
                                session_config=variant.session_config,
                            )
                            for variant in variants
                        },
                    }
                )

    comparison_summaries = {
        comparison_name: {
            "control": control,
            "candidate": candidate,
            "geometric_mean_latency_speedup": _geometric_mean(
                [case["comparisons"][comparison_name]["median_latency_speedup"] for case in cases]
            ),
            "minimum_case_speedup": min(
                case["comparisons"][comparison_name]["median_latency_speedup"] for case in cases
            ),
            "maximum_case_speedup": max(
                case["comparisons"][comparison_name]["median_latency_speedup"] for case in cases
            ),
        }
        for comparison_name, control, candidate in comparison_specs
    }
    tail_input = [
        {
            "batch_size": case["batch_size"],
            "sequence_length": case["sequence_length"],
            "baseline": case["variants"]["fp32_control"],
            "optimized": case["variants"]["qint8_control"],
            "process_cpu": {
                "baseline": case["variants"]["fp32_control"]["process_cpu"],
                "optimized": case["variants"]["qint8_control"]["process_cpu"],
            },
        }
        for case in cases
    ]
    result: dict[str, Any] = {
        "schema_version": "armbench-experiment/v1",
        "experiment": {
            "id": BF16_EXPERIMENT_ID,
            "parent": BF16_PARENT_EXPERIMENT,
            "origin": "agent-proposed-after-primary-source-review",
            "hypothesis": (
                "Enabling MLAS Arm64 BF16 fast-math reduces absolute latency for the FP32 "
                "model and/or the residual FP32 GEMMs in the dynamic-QInt8 model."
            ),
            "smallest_delta": (f"Set only the ONNX Runtime session entry {BF16_FASTMATH_KEY}=1."),
            "success_criteria": {
                "same_artifact_geometric_mean_speedup_at_least": 1.03,
                "minimum_case_speedup_at_least": 0.97,
                "mean_embedding_cosine_at_least": 0.99,
                "promotion_requires_independent_native_runs": 5,
            },
            "rejection_criterion": (
                "Reject as no material effect if both same-artifact geometric-mean speedups "
                "are at most 1.01; reject any candidate below the quality gate."
            ),
            "abort_criterion": "Abort when the native target does not advertise BF16 support.",
            "code_revision": code_revision or os.getenv("GITHUB_SHA") or "working-tree",
            "data_manifest": "not-applicable: generated fixed-shape inputs from authored text",
            "fold_map": "not-applicable: deterministic inference benchmark",
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": "Apache-2.0",
        },
        "configuration": {
            "batch_sizes": list(batch_sizes),
            "sequence_lengths": list(sequence_lengths),
            "warmups_per_variant_and_case": warmups,
            "discarded_warmups_per_variant_and_block": block_warmups,
            "measured_iterations_per_variant_and_case": iterations,
            "measurement_blocks_per_case": measurement_blocks,
            "random_seed": random_seed,
            "bootstrap_resamples": bootstrap_resamples,
            "intra_op_threads": threads,
            "inter_op_threads": 1,
            "profiled_inferences_per_variant_and_case": (
                profile_inferences if profile_dir is not None else 0
            ),
            "timing_boundary": (
                "ONNX inference plus mean pooling and L2 normalization; tokenization excluded"
            ),
        },
        "machine": _machine_metadata(),
        "variants": variant_metadata,
        "quality_by_variant": quality_by_variant,
        "cases": cases,
        "profiles": profiles,
        "summary": {
            "comparisons": comparison_summaries,
            "tail_spike_cases_for_incumbent_pair": _tail_spike_cases(tail_input),
        },
    }
    result["verdict"] = evaluate_bf16_verdict(result)
    return result


def render_bf16_markdown(result: Mapping[str, Any]) -> str:
    """Render a compact, reviewable experiment ledger entry."""

    experiment = result["experiment"]
    machine = result["machine"]
    lines = [
        f"# Experiment {experiment['id']}",
        "",
        f"- Generated: `{result['generated_at_utc']}`",
        f"- Parent: `{experiment['parent']}`",
        f"- Code revision: `{experiment['code_revision']}`",
        "",
        "## Contract",
        "",
        f"Hypothesis: {experiment['hypothesis']}",
        "",
        f"Smallest delta: `{BF16_FASTMATH_KEY}=1`.",
        "",
        "Success requires at least 1.03x same-artifact geometric-mean speedup, no shape "
        "below 0.97x, mean embedding cosine at least 0.99, and five independent native runs.",
        "",
        "## Environment",
        "",
        f"- Architecture: `{machine['architecture']}`",
        f"- ONNX Runtime: `{machine['onnxruntime']}`",
        f"- Threads: `{result['configuration']['intra_op_threads']}` intra-op / `1` inter-op",
        "",
        "## Per-shape result",
        "",
        (
            "| Batch | Seq | FP32 ms | FP32+BF16 ms | FP32 BF16 speedup | "
            "QInt8 ms | QInt8+BF16 ms | QInt8 BF16 speedup |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        variants = case["variants"]
        comparisons = case["comparisons"]
        lines.append(
            "| {batch} | {sequence} | {fp32:.3f} | {fp32_bf16:.3f} | {fp32_speedup:.3f}x "
            "| {qint8:.3f} | {qint8_bf16:.3f} | {qint8_speedup:.3f}x |".format(
                batch=case["batch_size"],
                sequence=case["sequence_length"],
                fp32=variants["fp32_control"]["median_ms"],
                fp32_bf16=variants["fp32_bf16_fastmath"]["median_ms"],
                fp32_speedup=comparisons["fp32_bf16_vs_control"]["median_latency_speedup"],
                qint8=variants["qint8_control"]["median_ms"],
                qint8_bf16=variants["qint8_bf16_fastmath"]["median_ms"],
                qint8_speedup=comparisons["qint8_bf16_vs_control"]["median_latency_speedup"],
            )
        )
    lines.extend(["", "## Aggregate and quality", ""])
    for name in ("fp32_bf16_vs_control", "qint8_bf16_vs_control"):
        summary = result["summary"]["comparisons"][name]
        lines.append(
            f"- `{name}`: {summary['geometric_mean_latency_speedup']:.3f}x geometric mean; "
            f"range {summary['minimum_case_speedup']:.3f}x to "
            f"{summary['maximum_case_speedup']:.3f}x."
        )
    for name in ("fp32_bf16_fastmath", "qint8_bf16_fastmath"):
        quality = result["quality_by_variant"][name]
        lines.append(
            f"- `{name}` mean cosine versus FP32 control: {quality['mean_embedding_cosine']:.8f}."
        )
    verdict = result["verdict"]
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"**{verdict['status']}** — {verdict['reason']}",
            "",
            "This experiment is independent evidence and does not replace the submitted "
            "2.45x headline without the declared promotion gate.",
            "",
        ]
    )
    return "\n".join(lines)


def write_bf16_experiment(result: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write machine-readable and reviewer-readable experiment evidence."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "experiment.json",
        "markdown": output_dir / "experiment.md",
    }
    paths["json"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    paths["markdown"].write_text(render_bf16_markdown(result), encoding="utf-8")
    return paths
