# Public release checklist

Release target: `https://github.com/yhay81/armbench-minilm`  
Challenge: Arm Create: AI Optimization Challenge, Cloud AI track  
Decision: public release is required for challenge submission and was requested by the repository owner.

- [x] Original project code has an OSI-approved license (MIT).
- [x] Source model revision and Apache-2.0 license are recorded.
- [x] Challenge rules and native Arm64 runner sources are recorded in `project.toml`.
- [x] Generated model binaries, caches, results, credentials, and tokens are gitignored.
- [x] No private, gated, competition-only, or personal data is used.
- [x] The workload text is authored and labeled as such.
- [x] Setup, reproduction, methodology, timing boundary, and limitations are documented.
- [x] Tests cover numerical helpers and report generation.
- [x] Repository-local quality gates pass from the lockfile (`ruff`, `ty`, 10 tests).

The source is approved for public release. The remaining checks are required before the Devpost submission is finalized:

- [x] Native Arm64 workflow completes: https://github.com/yhay81/armbench-minilm/actions/runs/31405378460
- [x] Measured results are copied to the README using the generated report precision.
- [x] Devpost draft links to the public repository and Arm64 workflow run.
- [x] A 74-second Full HD demo and reproducible Remotion source are committed; all performance and quality figures are read from retained benchmark JSON.
- [x] The demo uses no third-party music, stock footage, generated voice, Arm logo, or private data; its ambient bed is synthesized locally with FFmpeg.
- [ ] The owner uploads the verified MP4 to YouTube, Vimeo, or Youku and adds that public URL to Devpost.
- [x] Final Devpost preview contains no unsupported performance or impact claim.
- [x] Submitted project: https://devpost.com/software/armbench-minilm
