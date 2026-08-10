# Performance-to-the-limit roadmap

- Status: proposed on 2026-08-11
- Scope: native Arm64 CPU sentence-embedding inference

North star: approach the best latency, throughput, and model size that the selected hardware can deliver without hiding quality loss or measurement uncertainty.

## Execution status

| Stage | Status | Evidence |
|---|---|---|
| R0 implementation | Implemented and under native validation | Fixed shape grid, balanced-randomized A/B blocks, per-block warm-ups, pinned ORT spin behavior, wall/process-CPU samples, bootstrap intervals, hardware metadata, and separate ORT profiles |
| R0 exit gate | Pending | First three-run audit exposed short-input thread-pool variability; repeat after the protocol correction |
| R1–R7 | Planned | Start after the R0 exit gate is satisfied |

## What “the limit” means

There is no single universal speed limit. The limit is defined for a pinned tuple:

`model × precision × batch × sequence length × thread count × runtime build × Arm CPU`

For each tuple, ArmBench will estimate three bounds:

1. **Artifact-size bound** — initializer payload at the chosen precision, plus unavoidable graph and quantization metadata.
2. **Operator-scope bound** — Amdahl's-law limit if the currently targeted operators became infinitely fast.
3. **Hardware roofline bound** — the larger of compute time and memory-traffic time, plus measured non-kernel overhead.

The project will call a result “near-limit” only when it reports the bound, the measured value, the remaining gap, and the quality gate together.

## Current position

The submitted result is already strong: a 2.45x geometric-mean median-latency speedup, a 35.0% model-file reduction, and 0.99267173 mean FP32/INT8 embedding cosine on the authored workload.

An exact analysis of the pinned FP32 ONNX graph gives the following size budget:

Reproduce it after `prepare` with:

```bash
uv run armbench-minilm bounds --work-dir .armbench --output results/size-bounds.json
```

| Quantity | Payload |
|---|---:|
| FP32 initializers | 86.080 MiB |
| FP32 weights currently targeted by constant-weight `MatMul/Gemm` | 40.500 MiB |
| INT8 payload lower bound for those targeted weights | 10.125 MiB |
| Non-target FP32 initializers | 45.580 MiB |
| Current-scope theoretical initializer bound | **55.705 MiB** |
| Actual optimized ONNX file | **56.038 MiB** |

The actual file is only about **0.6% above the current-scope payload bound**. Therefore, more work on the same weight-only `MatMul/Gemm` conversion cannot materially improve model size. A smaller artifact requires quantizing additional tensors—especially the embedding table—or changing the model architecture.

If every current FP32 initializer could be stored without overhead, the payload-only bounds would be 21.520 MiB at 8 bits and 10.760 MiB at 4 bits. These are accounting bounds, not promises: scales, zero points, alignment, unsupported operators, and accuracy-preserving FP32 fallbacks add bytes.

The latency gap is not yet knowable. The current report lacks per-operator Arm64 profiles, fixed sequence-length cases, memory-bandwidth measurements, and repeated-run confidence intervals. Establishing those is the first priority.

## Non-negotiable measurement contract

Every promoted result must satisfy all of the following:

- Compare variants on the same clean machine and runtime build.
- Separate cold start, tokenization, encoder inference, pooling, normalization, and output-copy time.
- Benchmark a fixed `(batch, sequence length, threads)` grid; do not let natural sentence length silently change the shape.
- Interleave or randomize candidate and baseline trials to reduce thermal and VM-time bias.
- Preserve raw samples, not only median and p95 summaries.
- Use at least five independent process runs for promoted claims and report bootstrap confidence intervals.
- Record CPU identity and features, cache topology, governor/frequency information when available, runtime build flags, kernel version, and memory.
- Keep immutable evidence by commit and workflow-run identifier.
- Reject a speed result when the quality gate fails, even if the latency is excellent.

Initial stability gates:

- 95% confidence-interval half-width below 2% for median latency.
- Run-to-run coefficient of variation below 3% on the controlled target.
- No unexplained p95 spike above 1.5x the median. The submitted batch-8 INT8 run currently needs investigation because p95 is 14.299 ms versus a 7.376 ms median.

## Bound equations

For a model variant, record:

```text
size_scope_bound = non_target_bytes + target_elements × precision_bits / 8

operator_scope_speedup_limit =
    baseline_time / (baseline_time - time_in_targeted_operators)

roofline_time =
    max(integer_or_float_operations / sustainable_compute_rate,
        bytes_moved / sustainable_memory_bandwidth)
    + measured_non_kernel_time
```

The rates must be measured on the actual target, not copied from a marketing peak. Per-operator ONNX Runtime profiles identify `time_in_targeted_operators`; a small bandwidth microbenchmark and kernel microbenchmarks establish sustainable rates.

## Roadmap

### R0 — Make the benchmark scientifically stable

Target: first checkpoint; no optimization claim changes yet.

- [x] Add a fixed grid for batches `1, 8, 32` and sequence lengths `16, 32, 64, 128`.
- [ ] Add the thread-count dimension `1, 2, 4` after the primary four-thread protocol is stable.
- [x] Keep authored-text fidelity evaluation separate from the fixed-shape performance workload.
- [x] Store every wall-clock/process-CPU latency sample and trial order.
- [x] Run balanced-randomized A/B blocks rather than all FP32 measurements followed by all INT8 measurements.
- [x] Re-warm each model immediately before its measured block and pin ORT's intra-op spin duration/backoff; retain every timed single-inference sample.
- [ ] Complete three independent workflow repeats; deterministic bootstrap confidence intervals are implemented.
- [x] Capture relevant `/proc/cpuinfo` flags, cache topology, ONNX Runtime build/version, and runner image metadata.
- [x] Enable separate-session ONNX Runtime profiling and summarize time by operator and node.

Exit gate: the same commit produces stable results in three consecutive Arm64 workflows and explains the existing batch-8 tail-latency spike.

### R1 — Build the roofline and Amdahl model

Target: convert “2.45x faster” into “X% of the measurable ceiling.”

- Count per-node operations and bytes for each fixed shape.
- Measure sustainable single- and four-thread memory bandwidth.
- Microbenchmark the relevant FP32, dynamic-INT8, static-INT8, and later INT4 matrix shapes.
- Calculate the baseline time share in quantizable matrix operations, shape operations, pooling/normalization, allocation, and Python/runtime overhead.
- Publish the operator-scope infinite-speedup bound and hardware roofline gap for every grid point.
- Extend the existing `bounds` command so each experimental model records its own precision scope and candidate gap.

Exit gate: every headline latency has a reproducible lower-bound estimate and a named dominant bottleneck.

### R2 — Remove graph and framework overhead

Target: close the non-kernel gap before changing precision again.

- Run ONNX Runtime symbolic shape inference and transformer optimization before quantization; verify attention, GELU, layer-normalization, and skip-layer-normalization fusions.
- Produce fixed-shape optimized graphs for the most important serving shapes while retaining a dynamic-shape portability variant.
- Move attention-mask mean pooling and L2 normalization into the ONNX graph to avoid copying token embeddings into NumPy.
- Reuse input/output buffers and test I/O binding or equivalent preallocation.
- Compare the Python harness with `onnxruntime_perf_test` or a minimal C++ harness to isolate language/runtime overhead.
- Sweep intra-op threads, affinity, spinning, and the ONNX Runtime dynamic thread-pool cost model.

Exit gate: remaining fixed overhead is measured, and the selected graph produces numerically equivalent normalized embeddings.

### R3 — Find the INT8 Pareto frontier

Target: determine whether activation quantization beats the current dynamic weight-only path.

- Compare the current dynamic per-channel signed INT8 model against static S8S8 QDQ and QOperator candidates.
- Test MinMax, Entropy, and Percentile calibration on a pinned, representative calibration set.
- Optimize the transformer graph before quantization so transformer-specific quantized fusions can match.
- Use ONNX Runtime quantization debugging to identify sensitive weights and activations; keep only the minimum FP32 fallback set needed for quality.
- Measure conversion overhead, session-load time, latency, throughput, memory, size, and task quality for every candidate.

Exit gate: retain only non-dominated variants. Promote a new default only after repeated native Arm64 wins at the target shapes.

### R4 — Reach the Arm kernel ceiling

Target: prove which Arm instructions and micro-kernels produce the win.

- Identify whether the target exposes DotProd, I8MM, SVE/SVE2, or SME/SME2.
- Compare the pinned official wheel with a reproducible architecture-tuned ONNX Runtime build.
- Verify KleidiAI/MLAS kernel use; A/B test the ONNX Runtime `mlas.disable_kleidiai` option where supported.
- Benchmark matching KleidiAI matrix micro-kernels directly to separate kernel efficiency from graph/runtime overhead.
- On a controlled Arm instance, collect hardware counters for cycles, instructions, cache misses, and bandwidth when permissions allow.
- Do not combine measurements from different Arm CPUs into one headline.

Exit gate: the selected runtime is within 10% of the best measured kernel/runtime configuration, or the remaining gap is attributed to a specific operator or data movement.

### R5 — Cross the 8-bit boundary

Target: explore the next size and bandwidth frontier without disguising quality loss.

- Evaluate ONNX Runtime `MatMulNBits` INT4/UINT4 variants across supported block sizes and RTN, HQQ, or GPTQ algorithms.
- Test mixed W4/W8/FP32 assignments based on layer sensitivity.
- Quantize or compress the large embedding table; measure dequantization and gather costs rather than assuming smaller is faster.
- Match candidates to Arm KleidiAI INT4-capable micro-kernels when the target architecture supports them.
- Compare W4A16, W4A8, and weight-only paths on the complete `(batch, sequence length)` grid.

Exit gate: a candidate must improve at least one of latency, throughput, or size without being worse on all others and without violating the task-quality gate.

### R6 — Change the architecture when kernel optimization saturates

Target: move the true frontier after same-model optimization is exhausted.

- Distill six layers into a smaller encoder and evaluate four-, three-, and two-layer candidates.
- Test structured width and feed-forward pruning rather than unstructured sparsity that the runtime cannot accelerate.
- Evaluate vocabulary/embedding-table compression and smaller output dimensions such as 256 or 128.
- Train with quantization awareness only if post-training quantization cannot meet the quality gate.
- Treat each architecture as a separately identified model; never compare it as if it were bitwise-equivalent optimization.

Exit gate: publish a latency–size–quality Pareto curve, not a single cherry-picked winner.

### R7 — Validate the hardware-specific limit

Target: replace an anonymous VM ceiling with defensible Arm platform ceilings.

- Repeat the final matrix on explicitly identified Arm cloud CPUs from more than one generation.
- Report single-request latency, per-core throughput, scale-out throughput, and energy only where trustworthy telemetry exists.
- Keep the GitHub-hosted runner as the reproducible public baseline, not as a claim about every Arm server.

Exit gate: each platform has its own bound, measured gap, and reproducible configuration.

## Quality gates

The authored 32-sentence cosine check remains a fast regression test, but it is not enough for R3–R6. Add a pinned, license-reviewed MTEB subset covering semantic textual similarity and retrieval. For each candidate, require:

- mean corresponding-embedding cosine at least 0.99 unless a task benchmark justifies a different trade-off;
- no more than 0.5 absolute points of loss on the pinned STS primary score;
- no more than 1% relative loss on the pinned retrieval primary score;
- identical tokenizer, truncation, pooling, and normalization semantics;
- a documented exception for every FP32 fallback or excluded operator.

Thresholds are initial engineering gates, not claims that those differences are universally harmless.

## Challenge sprint order

Before the current submission deadline, prioritize work that strengthens evidence without replacing a good result with a risky one:

1. **R0:** fixed shapes, raw samples, repeated A/B trials, and hardware metadata.
2. **R1:** operator profile, size accounting, and first Amdahl/roofline chart.
3. **R2:** transformer graph optimization plus fused pooling/normalization.
4. **R3:** one carefully calibrated static S8S8 comparison.
5. **R4:** verify whether KleidiAI kernels are active on the public runner.

R5 and R6 should remain experimental until their quality evidence is complete. The submitted 2.45x run stays immutable; a new result replaces the headline only after passing all gates.

## Primary references

- [ONNX Runtime quantization documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [ONNX Runtime transformer optimization](https://onnxruntime.ai/docs/performance/transformers-optimization.html)
- [ONNX Runtime performance troubleshooting](https://onnxruntime.ai/docs/performance/tune-performance/troubleshooting.html)
- [ONNX Runtime session configuration keys](https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)
- [Arm KleidiAI](https://github.com/ARM-software/kleidiai)
- [Arm ONNX Runtime and KleidiAI learning path](https://learn.arm.com/learning-paths/mobile-graphics-and-gaming/performance_onnxruntime_kleidiai_sme2/overview/)
- [GitHub-hosted runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
- [MTEB official repository](https://github.com/embeddings-benchmark/mteb)
