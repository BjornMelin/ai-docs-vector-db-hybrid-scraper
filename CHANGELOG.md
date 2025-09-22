# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Introduced `.github/dependabot.yml` to automate weekly updates for GitHub Actions and Python dependencies.
- Documented CI branch-protection guidance and pinned action examples across developer and security guides.

### Changed
- Consolidated CI into a lean `core-ci.yml` pipeline (lint → tests with coverage → build → dependency audit) and introduced on-demand security and regression workflows while deleting `fast-feedback.yml`, `status-dashboard.yml`, and other scheduled automation.
- Simplified documentation checks to rely on `scripts/dev.py validate --check-docs --strict` plus strict MkDocs builds with pinned docs extras.
- Documented manual triggers for the security and regression workflows in `CONTRIBUTING.md` so contributors can opt into deeper validation without slowing default CI.
- Standardized workflow environment setup on the shared `.github/actions/setup-environment` composite and ensured all referenced actions remain pinned to immutable SHAs.
- Retired the SARIF upload path in favor of `pip-audit`, `safety`, and `bandit` reports stored as artifacts for manual review when the security workflow runs.

### Security
- Applied SHA pinning across composite actions and documentation snippets, aligning with GitHub’s secure-use guidance to mitigate supply-chain risk.

[Unreleased]: https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/compare/main...HEAD
