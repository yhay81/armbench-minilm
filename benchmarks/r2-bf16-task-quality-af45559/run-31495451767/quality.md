# Experiment r2-bf16-task-quality-v1

- Generated: `2026-08-11T13:27:56.427512+00:00`
- Parent: `r2-bf16-fastmath-v1`
- Code revision: `af45559851b8a500c02df8155a44bd8b8f76b9a1`
- Architecture: `aarch64`

## Fixed task contract

- IndicCrosslingualSTS: 12 x 256 pairs, macro cosine Spearman x 100.
- ArguAna: 8,674 documents and 1,406 queries, nDCG@10 x 100.
- ArguAna source limitation: 5 pinned qrel targets are absent from the corpus and therefore score zero.
- Tokenizer/model revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- This is a max-length-128 engineering gate, not an official MTEB leaderboard result.

## Task scores

| Variant | STS macro Spearman | ArguAna nDCG@10 |
|---|---:|---:|
| `fp32_control` | -6.2773 | 48.9917 |
| `fp32_bf16_fastmath` | -6.2677 | 49.0337 |
| `qint8_control` | -5.4882 | 49.0047 |
| `qint8_bf16_fastmath` | -5.7942 | 48.7270 |

## Predeclared comparison gates

| Comparison | STS loss (points) | Retrieval relative loss | STS cosine | Retrieval cosine | Verdict |
|---|---:|---:|---:|---:|---|
| `fp32_bf16_vs_control` | -0.0096 | -0.0856% | 0.99998246 | 0.99998768 | **passed** |
| `qint8_bf16_vs_control` | 0.3059 | 0.5666% | 0.99025232 | 0.99515090 | **passed** |
| `qint8_control_vs_fp32` | -0.7890 | -0.0265% | 0.98911148 | 0.99431392 | **rejected-task-quality-gate** |
| `qint8_bf16_vs_fp32` | -0.4831 | 0.5403% | 0.98910185 | 0.99432262 | **rejected-task-quality-gate** |

## STS slices

| Language pair | FP32 | FP32+BF16 | QInt8 | QInt8+BF16 |
|---|---:|---:|---:|---:|
| `en-as` | -18.9336 | -18.8601 | -17.8980 | -18.4704 |
| `en-bn` | -14.6349 | -14.7292 | -14.5233 | -14.5166 |
| `en-gu` | 5.3156 | 5.4431 | 5.1906 | 3.8559 |
| `en-hi` | -8.3639 | -8.3624 | -9.6406 | -9.0152 |
| `en-kn` | -6.4004 | -6.4462 | -5.0006 | -5.8120 |
| `en-ml` | 8.0847 | 8.0954 | 9.7037 | 9.1014 |
| `en-mr` | -6.4925 | -6.5008 | -5.9615 | -5.5044 |
| `en-or` | -4.8858 | -4.9700 | -1.5900 | -4.0133 |
| `en-pa` | -0.0342 | -0.0415 | 1.3872 | 2.1415 |
| `en-ta` | -20.9817 | -21.0428 | -20.6091 | -20.8202 |
| `en-te` | 0.2198 | 0.3138 | 1.4721 | 1.0583 |
| `en-ur` | -8.2204 | -8.1117 | -8.3892 | -7.5349 |

## Verdict

**fp32-bf16-task-quality-gate-passed** — FP32+BF16 passed the task gate; its five-run performance gate was completed separately. QInt8 conclusions remain limited to their recorded comparisons.

The data files are fetched from exact official MTEB task revisions, hash-checked, and not redistributed in this repository.
