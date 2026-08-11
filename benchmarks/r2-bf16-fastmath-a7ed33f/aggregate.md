# BF16 repeated-run analysis

- Experiment: `r2-bf16-fastmath-v1`
- Analysis code: `feaa1f9cefc183f4b9c9a554a7c9235cec374549`
- Independent native runs: **5**
- Decision: **performance-repetition-gate-passed-needs-task-quality**

## Run-level result

| Run | Revision | FP32 + BF16 GM | FP32 minimum | QInt8 + BF16 GM | QInt8 minimum |
|---:|---|---:|---:|---:|---:|
| [31485130635](https://github.com/yhay81/armbench-minilm/actions/runs/31485130635) | `a7ed33f` | 1.3328x | 1.3145x | 1.0194x | 1.0020x |
| [31486949258](https://github.com/yhay81/armbench-minilm/actions/runs/31486949258) | `da709ab` | 1.3248x | 1.3001x | 1.0176x | 1.0020x |
| [31486966714](https://github.com/yhay81/armbench-minilm/actions/runs/31486966714) | `da709ab` | 1.3348x | 1.3052x | 1.0219x | 1.0085x |
| [31486977343](https://github.com/yhay81/armbench-minilm/actions/runs/31486977343) | `da709ab` | 1.3384x | 1.3100x | 1.0176x | 0.9909x |
| [31486986690](https://github.com/yhay81/armbench-minilm/actions/runs/31486986690) | `da709ab` | 1.3386x | 1.3122x | 1.0212x | 1.0036x |

## Shape stability

| Batch | Seq | FP32 + BF16 median | CV | QInt8 + BF16 median | CV |
|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 1.3608x | 1.96% | 1.0097x | 0.90% |
| 1 | 32 | 1.3436x | 1.35% | 1.0177x | 0.66% |
| 1 | 64 | 1.3397x | 1.27% | 1.0222x | 0.49% |
| 1 | 128 | 1.3281x | 1.22% | 1.0311x | 0.96% |
| 8 | 16 | 1.3272x | 0.58% | 1.0097x | 0.82% |
| 8 | 32 | 1.3293x | 0.36% | 1.0164x | 0.49% |
| 8 | 64 | 1.3413x | 0.18% | 1.0208x | 0.75% |
| 8 | 128 | 1.3277x | 0.41% | 1.0354x | 0.42% |
| 32 | 16 | 1.3436x | 0.11% | 1.0062x | 0.23% |
| 32 | 32 | 1.3356x | 0.25% | 1.0096x | 0.46% |
| 32 | 64 | 1.3311x | 0.16% | 1.0231x | 0.24% |
| 32 | 128 | 1.3122x | 0.41% | 1.0375x | 0.08% |

## Decision

FP32 + BF16 run-level geometric-mean speedup: median **1.3348x**, CV **0.38%**.

QInt8 + BF16 run-level geometric-mean speedup: median **1.0194x**, CV **0.18%**.

- FP32 + BF16: **retain-for-pinned-sts-and-retrieval-gate**
- QInt8 + BF16: **shape-specific-follow-up**
- Default/headline promotion: **blocked on pinned STS and retrieval quality**
