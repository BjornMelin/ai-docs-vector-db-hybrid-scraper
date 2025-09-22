# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Introduced `.github/dependabot.yml` to automate weekly updates for GitHub Actions and Python dependencies.
- Documented CI branch-protection guidance and pinned action examples across developer and security guides.

### Changed
- Consolidated CI into `core-ci.yml` with dedicated docs, security, and regression opt-in workflows; removed redundant fast-feedback, status dashboard, and schedule-based automation to enforce KISS/DRY principles.
- Simplified documentation pipeline to rely on `scripts/dev.py validate` and MkDocs with pinned docs extras only.
- Replaced all GitHub Actions references with immutable commit SHAs (including first-party actions) and pinned `setup-uv` to version `0.8.19` for deterministic builds.
- Refactored the CI workflow to use a fan-in `ci-gate` job, ensuring branch protection relies on a single aggregated status and enforcing an 80% coverage threshold consistently.
- Tuned `tests/ci/performance_reporter.py` for precise psutil sampling, UTC timestamps, and Ruff-compliant patterns while reducing per-test overhead.
- Hardened container security scans by pinning Trivy/Hadolint actions, fixing Dockerfile selection, and locking the Trivy CLI version.
- Standardized workflow environment setup on the shared `.github/actions/setup-environment` composite, removed `PYTHONOPTIMIZE`, and aligned Python/uv versions across CI and automation jobs.
- Replaced legacy benchmark and documentation scripts with the unified `scripts/dev.py` CLI and updated workflow/documentation references accordingly.
- Simplified the security job to emit SARIF reports via `pip-audit` and `bandit`, uploading directly to GitHub code scanning without auxiliary processing steps.

### Security
- Applied SHA pinning across composite actions and documentation snippets, aligning with GitHub’s secure-use guidance to mitigate supply-chain risk.

[Unreleased]: https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/compare/main...HEAD
