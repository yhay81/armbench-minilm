# Passing R0 validation for `21c3d6f`

The corrected schema-v2 protocol passed the R0 stability gate in three
consecutive public native Arm64 workflows. This validates the measurement
foundation; it does not replace the immutable challenge-submission headline.

## Immutable runs

- Commit: `21c3d6f07499074aae353f085316673bcc649daf`
- CPU part: `0xd49`
- Relevant features: `asimd`, `i8mm`, `sve`, `sve2`, `svei8mm`
- Runner image: `ubuntu24-arm64` version `20260719.67.1`
- ONNX Runtime: `1.28.0`, `CPUExecutionProvider`
- [Run 31414573815](https://github.com/yhay81/armbench-minilm/actions/runs/31414573815)
- [Run 31414596895](https://github.com/yhay81/armbench-minilm/actions/runs/31414596895)
- [Run 31414600562](https://github.com/yhay81/armbench-minilm/actions/runs/31414600562)

Each run directory below preserves the machine-readable result with every timed
sample and the generated Markdown report. Raw ORT trace files remain attached to
the public workflow runs; their operator summaries are also embedded in each
`benchmark.json`.

## Run-level result

| Run | Fixed-grid geometric-mean speedup | Maximum median 95% CI half-width | p95 cases above 1.5x |
|---:|---:|---:|---:|
| 31414573815 | 2.67254x | 0.939% | 0 |
| 31414596895 | 2.68750x | 0.658% | 0 |
| 31414600562 | 2.67074x | 0.827% | 0 |

Across all 12 controlled shapes, the maximum run-to-run CV was 2.356% for the
FP32 median, 2.669% for the INT8 median, and 1.119% for speedup. The largest
observed wall-clock p95/median ratio was 1.260. All are inside the initial R0
limits of 2% CI half-width, 3% run-to-run CV, and no unexplained p95 above 1.5x.

## Three-run shape aggregate

| Batch | Sequence | FP32 median (ms) | INT8 median (ms) | Mean speedup | Speedup CV |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 2.9367 | 1.5941 | 1.8423x | 0.351% |
| 1 | 32 | 5.0598 | 2.4714 | 2.0475x | 0.437% |
| 1 | 64 | 9.3690 | 3.9383 | 2.3792x | 1.119% |
| 1 | 128 | 18.4663 | 7.5489 | 2.4462x | 0.136% |
| 8 | 16 | 17.6168 | 6.6974 | 2.6305x | 0.768% |
| 8 | 32 | 33.2038 | 11.1345 | 2.9824x | 0.739% |
| 8 | 64 | 66.8076 | 21.9742 | 3.0403x | 0.307% |
| 8 | 128 | 135.8831 | 47.3935 | 2.8671x | 0.253% |
| 32 | 16 | 65.8412 | 21.0238 | 3.1318x | 0.250% |
| 32 | 32 | 128.9774 | 40.5709 | 3.1791x | 0.241% |
| 32 | 64 | 258.3363 | 82.7304 | 3.1226x | 0.141% |
| 32 | 128 | 533.8811 | 184.0275 | 2.9011x | 0.235% |

The previous batch-8 tail spike is resolved without deleting a timed sample.
For batch 8 / sequence 16, the maximum wall p95/median ratio fell from 1.815 in
the failed audit to 1.203, while its process-CPU ratio was at most 1.093. The
correction was symmetric per-block re-warming plus pinned time-bounded ORT spin
behavior.

Model size and fidelity were unchanged: 35.004% file-size reduction,
0.99267173 mean embedding cosine, and 0.97647780 minimum embedding cosine.
