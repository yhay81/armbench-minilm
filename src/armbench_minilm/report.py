"""Human-readable reports generated from benchmark JSON."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _bytes_to_mib(value: int) -> float:
    return value / (1024 * 1024)


def _format_median(metrics: dict[str, Any]) -> str:
    median = metrics["median_ms"]
    low = metrics.get("median_ci95_low_ms")
    high = metrics.get("median_ci95_high_ms")
    if low is None or high is None:
        return f"{median:.3f}"
    return f"{median:.3f} [{low:.3f}, {high:.3f}]"


def _format_speedup(item: dict[str, Any]) -> str:
    speedup = item["median_latency_speedup"]
    low = item.get("speedup_ci95_low")
    high = item.get("speedup_ci95_high")
    if low is None or high is None:
        return f"{speedup:.2f}x"
    return f"{speedup:.2f}x [{low:.2f}, {high:.2f}]"


def render_markdown(result: dict[str, Any]) -> str:
    """Render one benchmark result as a portable Markdown report."""

    machine = result["machine"]
    models = result["models"]
    quality = result["quality"]
    summary = result["summary"]
    likely_host_spikes = sum(
        case.get("classification") == "likely_vm_preemption_or_host_contention"
        for case in summary.get("tail_spike_cases", [])
    )
    cpu_cores = (
        f"{machine['physical_cpu_cores']} physical / "
        f"{machine['logical_cpu_cores']} logical"
    )
    baseline_size = _bytes_to_mib(models["baseline"]["size_bytes"])
    optimized_size = _bytes_to_mib(models["optimized"]["size_bytes"])
    baseline_sha = models["baseline"]["sha256"]
    optimized_sha = models["optimized"]["sha256"]
    configuration = result["configuration"]
    linux_cpu = machine.get("linux_cpu") or {}
    relevant_arm_features = [
        feature
        for feature in linux_cpu.get("features", [])
        if feature in {"asimd", "dotprod", "i8mm", "sve", "sve2", "sme", "sme2"}
    ]
    lines = [
        "# ArmBench MiniLM result",
        "",
        f"Generated: `{result['generated_at_utc']}`",
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Architecture | `{machine['architecture']}` |",
        f"| OS | `{machine['operating_system']}` |",
        f"| CPU | `{machine['processor']}` |",
        f"| CPU cores | {cpu_cores} |",
        f"| ONNX Runtime | `{machine['onnxruntime']}` |",
        f"| Python | `{machine['python']}` |",
        f"| Threads | {result['configuration']['intra_op_threads']} intra-op / 1 inter-op |",
    ]
    if relevant_arm_features:
        lines.append(f"| Relevant Arm features | `{', '.join(relevant_arm_features)}` |")
    if result.get("schema_version", 1) >= 2:
        lines.extend(
            [
                "",
                "## Measurement protocol",
                "",
                (
                    f"Fixed sequence lengths: `{configuration['sequence_lengths']}`; "
                    f"{configuration['measurement_blocks_per_case']} balanced-randomized A/B "
                    "blocks; "
                    f"{configuration.get('discarded_warmups_per_model_and_block', 0)} discarded "
                    "warm-ups immediately before each model block; "
                    f"{configuration['measured_iterations_per_model_and_batch']} raw samples "
                    "per model and case."
                ),
                "",
                (
                    f"Median intervals use {configuration['bootstrap_resamples']} deterministic "
                    "bootstrap resamples. Wall-clock and process-CPU samples are both retained. "
                    "Each timed sample is one inference; no outlier is discarded. "
                    "Tokenization is excluded; timed samples include ONNX inference, mean pooling, "
                    "and L2 normalization."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Model size",
            "",
            "| Model | Format | Size (MiB) | SHA-256 |",
            "|---|---|---:|---|",
            f"| Baseline | FP32 | {baseline_size:.2f} | `{baseline_sha}` |",
            f"| Optimized | QInt8 | {optimized_size:.2f} | `{optimized_sha}` |",
            "",
            f"**Measured size reduction: {summary['model_size_reduction_percent']:.1f}%.**",
            "",
            "## Inference performance",
            "",
        ]
    )
    has_sequence_lengths = any("sequence_length" in item for item in result["batches"])
    if has_sequence_lengths:
        lines.extend(
            [
                (
                    "| Batch | Seq | FP32 median ms [95% CI] | INT8 median ms [95% CI] "
                    "| INT8 p95 ms | Speedup [95% CI] | INT8 sentences/s |"
                ),
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                (
                    "| Batch | FP32 median (ms) | INT8 median (ms) | INT8 p95 (ms) "
                    "| Speedup | INT8 sentences/s |"
                ),
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
    for item in result["batches"]:
        if has_sequence_lengths:
            lines.append(
                (
                    "| {batch_size} | {sequence_length} | {baseline} | {optimized} | {p95:.3f} "
                    "| {speedup} | {throughput:.1f} |"
                ).format(
                    batch_size=item["batch_size"],
                    sequence_length=item["sequence_length"],
                    baseline=_format_median(item["baseline"]),
                    optimized=_format_median(item["optimized"]),
                    p95=item["optimized"]["p95_ms"],
                    speedup=_format_speedup(item),
                    throughput=item["optimized"]["sentences_per_second"],
                )
            )
        else:
            lines.append(
                (
                    "| {batch_size} | {baseline:.3f} | {optimized:.3f} | {p95:.3f} "
                    "| {speedup:.2f}x | {throughput:.1f} |"
                ).format(
                    batch_size=item["batch_size"],
                    baseline=item["baseline"]["median_ms"],
                    optimized=item["optimized"]["median_ms"],
                    p95=item["optimized"]["p95_ms"],
                    speedup=item["median_latency_speedup"],
                    throughput=item["optimized"]["sentences_per_second"],
                )
            )
    lines.extend(
        [
            "",
            (
                "**Geometric-mean median-latency speedup: "
                f"{summary['geometric_mean_latency_speedup']:.2f}x.**"
            ),
        ]
    )
    if "maximum_median_ci95_half_width_percent" in summary:
        lines.extend(
            [
                "",
                (
                    "Maximum median-latency 95% CI half-width: "
                    f"**{summary['maximum_median_ci95_half_width_percent']:.2f}%**. "
                    f"Tail-spike cases above 1.5x p95/median: "
                    f"**{len(summary['tail_spike_cases'])}**; likely VM preemption or host "
                    f"contention: **{likely_host_spikes}**."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Fidelity guardrail",
            "",
            (
                f"Compared {result['configuration']['quality_sentence_count']} authored sentences "
                "using normalized embeddings."
            ),
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Mean FP32/INT8 embedding cosine | {quality['mean_embedding_cosine']:.8f} |",
            f"| Minimum FP32/INT8 embedding cosine | {quality['minimum_embedding_cosine']:.8f} |",
            (
                "| Mean pairwise-similarity absolute error | "
                f"{quality['mean_pairwise_similarity_error']:.8f} |"
            ),
            (
                "| Maximum pairwise-similarity absolute error | "
                f"{quality['maximum_pairwise_similarity_error']:.8f} |"
            ),
        ]
    )
    profiles = result.get("profiles", [])
    if profiles:
        canonical_profile = max(
            profiles,
            key=lambda item: (item["batch_size"], item["sequence_length"]),
        )
        lines.extend(
            [
                "",
                "## Operator profile",
                "",
                (
                    f"Canonical profile: batch {canonical_profile['batch_size']}, "
                    f"sequence length {canonical_profile['sequence_length']}."
                ),
                "",
                "| Model | Top operator | Profiled node time (µs) | Calls |",
                "|---|---|---:|---:|",
            ]
        )
        for model_name in ("baseline", "optimized"):
            for operator in canonical_profile[model_name]["operators"][:5]:
                lines.append(
                    (
                        "| {model} | `{operator}` | {duration:.0f} | {calls} |"
                    ).format(
                        model=model_name,
                        operator=operator["operator"],
                        duration=operator["duration_us"],
                        calls=operator["calls"],
                    )
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "These measurements describe one pinned model, workload, runtime, and machine. "
                "They are not a universal Arm64 performance claim. Re-run the workflow on the "
                "target hardware before making deployment decisions."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_html(markdown_report: str, result: dict[str, Any]) -> str:
    """Render a dependency-free HTML summary suitable for an artifact preview."""

    has_sequence_lengths = any("sequence_length" in item for item in result["batches"])
    rows = []
    for item in result["batches"]:
        sequence_cell = (
            f"<td>{item['sequence_length']}</td>" if has_sequence_lengths else ""
        )
        rows.append(
            (
                "<tr><td>{}</td>{}<td>{:.3f}</td><td>{:.3f}</td>"
                "<td>{:.2f}x</td><td>{:.1f}</td></tr>"
            ).format(
                item["batch_size"],
                sequence_cell,
                item["baseline"]["median_ms"],
                item["optimized"]["median_ms"],
                item["median_latency_speedup"],
                item["optimized"]["sentences_per_second"],
            )
        )
    machine = result["machine"]
    summary = result["summary"]
    quality = result["quality"]
    architecture = html.escape(str(machine["architecture"]))
    headline_speedup = summary["geometric_mean_latency_speedup"]
    size_reduction = summary["model_size_reduction_percent"]
    mean_cosine = quality["mean_embedding_cosine"]
    escaped_markdown = html.escape(markdown_report)
    sequence_header = "<th>Sequence</th>" if has_sequence_lengths else ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ArmBench MiniLM result</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    body {{ margin: 0; background: #07111f; color: #eaf2ff; }}
    main {{ max-width: 960px; margin: auto; padding: 48px 24px 72px; }}
    h1 {{ font-size: clamp(2rem, 6vw, 4.5rem); margin: 0; letter-spacing: -.04em; }}
    .tag {{ color: #8bd3ff; font-weight: 700; text-transform: uppercase; letter-spacing: .12em; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr));
      gap: 16px; margin: 32px 0; }}
    .card {{ padding: 22px; border: 1px solid #294461; border-radius: 16px; background: #0d1c30; }}
    .value {{ display: block; font-size: 2.25rem; color: #61e6b0; font-weight: 800; }}
    table {{ width: 100%; border-collapse: collapse; margin: 24px 0; }}
    th, td {{ text-align: right; padding: 12px; border-bottom: 1px solid #294461; }}
    th:first-child, td:first-child {{ text-align: left; }}
    code {{ color: #8bd3ff; }}
    details {{ margin-top: 40px; }}
    pre {{ white-space: pre-wrap; background: #0d1c30; padding: 20px;
      border-radius: 12px; overflow-wrap: anywhere; }}
  </style>
</head>
<body><main>
  <p class="tag">Native Arm64 · ONNX Runtime · Reproducible</p>
  <h1>ArmBench MiniLM</h1>
  <p>FP32-to-INT8 sentence embedding optimization measured on
    <code>{architecture}</code>.</p>
  <section class="cards">
    <div class="card"><span class="value">{headline_speedup:.2f}x</span>
      geometric-mean latency speedup</div>
    <div class="card"><span class="value">{size_reduction:.1f}%</span>
      model size reduction</div>
    <div class="card"><span class="value">{mean_cosine:.6f}</span>
      mean embedding cosine</div>
  </section>
  <h2>Performance by batch</h2>
  <table><thead><tr><th>Batch</th>{sequence_header}<th>FP32 median ms</th><th>INT8 median ms</th>
    <th>Speedup</th><th>INT8 sentences/s</th></tr></thead>
  <tbody>{''.join(rows)}</tbody></table>
  <p>Timing excludes tokenization and includes ONNX inference, mean pooling,
    and L2 normalization.</p>
  <details><summary>Full machine-readable narrative</summary>
    <pre>{escaped_markdown}</pre></details>
</main></body></html>
"""


def write_reports(result: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write JSON, Markdown, and standalone HTML representations."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_report = render_markdown(result)
    paths = {
        "json": output_dir / "benchmark.json",
        "markdown": output_dir / "report.md",
        "html": output_dir / "report.html",
    }
    paths["json"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    paths["markdown"].write_text(markdown_report, encoding="utf-8")
    paths["html"].write_text(render_html(markdown_report, result), encoding="utf-8")
    return paths
