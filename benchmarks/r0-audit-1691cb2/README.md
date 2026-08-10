# R0 three-run stability audit for `1691cb2`

This audit intentionally records an unsuccessful stability gate. It is evidence
for the measurement-protocol correction, not a promoted performance claim.

## Immutable inputs

All three public native Arm64 workflows measured the same commit and completed
successfully on the same runner image and CPU identity:

- Commit: `1691cb2c6fb5d43a02ddf8e1feed54e7c0e3f7d0`
- CPU part: `0xd49`
- Relevant features: `asimd`, `i8mm`, `sve`, `sve2`, `svei8mm`
- Runner image: `ubuntu24-arm64` version `20260719.67.1`
- [Run 31412846376](https://github.com/yhay81/armbench-minilm/actions/runs/31412846376)
- [Run 31412854829](https://github.com/yhay81/armbench-minilm/actions/runs/31412854829)
- [Run 31412863418](https://github.com/yhay81/armbench-minilm/actions/runs/31412863418)

The workflow artifacts retain all 100 single-inference wall-clock and process-CPU
samples per model and shape, block order, hardware metadata, and separate ORT
profiles.

## Gate result

| Run | Fixed-grid geometric-mean speedup | Maximum median 95% CI half-width | Tail classification count |
|---:|---:|---:|---|
| 31412846376 | 2.70537x | 1.183% | 5 in-process |
| 31412854829 | 2.71102x | **17.206%** | 5 in-process |
| 31412863418 | 2.69197x | 1.635% | 6 in-process, 1 likely host/VM |

The geometric-mean speedup was repeatable, but the strict R0 gate did not pass.
Batch 1, sequence 16 had 12.217% FP32 and 8.770% INT8 run-to-run CV; its speedup
CV was 3.483%. Batch 1, sequence 32 had 3.903% speedup CV. Larger shapes were
substantially more stable: every batch-8 or batch-32 speedup CV was below 1.52%.

The submitted batch-8 tail spike is now attributable rather than mysterious. In
the fixed grid, batch 8 / sequence 16 reached a 1.815 wall p95/median ratio and a
1.709 process-CPU p95/median ratio. Because both clocks moved together, the
diagnostic class is in-process variability, not VM descheduling. Longer batch-8
shapes had no tail case above 1.5x.

Inspection of raw block samples showed spikes after switching between the two
independent ONNX Runtime session thread pools. The correction therefore adds
symmetric discarded warm-ups immediately before each measured model block and
pins ORT's time-bounded intra-op spin behavior. Timed outliers remain untouched.
A new three-run audit is required before R0 can close.
