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

- [x] Native Arm64 workflow completes; final 100-iteration run will be recorded.
- [ ] Measured results are copied to the README without changing units or precision.
- [ ] Devpost description links to the public repository and Arm64 workflow run.
- [ ] Demo video shows the actual repository, workflow, and generated report.
- [ ] Final Devpost preview contains no unsupported performance or impact claim.
