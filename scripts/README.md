# Developer Scripts

The historical `scripts/` directory contained dozens of one-off automation files
that were created during the modernization of this project. Maintaining those
single-purpose scripts became expensive and many of them no longer reflected the
current architecture. The directory has been consolidated into a single
Python-based CLI that covers the supported workflows without additional wrappers.

## Primary entrypoint

```bash
python scripts/dev.py --help
```

The CLI exposes subcommands for the day-to-day workflows:

| Task | Command |
| --- | --- |
| Run quick unit tests | `python scripts/dev.py test --profile quick` |
| Run integration tests | `python scripts/dev.py test --profile integration` |
| Full suite with coverage | `python scripts/dev.py test --profile full --coverage` |
| Performance benchmarks | `python scripts/dev.py benchmark` |
| Start Qdrant services | `python scripts/dev.py services start` |
| Start monitoring stack | `python scripts/dev.py services start --stack monitoring` |
| Validate configuration and docs | `python scripts/dev.py validate --check-docs` |
| Run lint/format/typecheck | `python scripts/dev.py quality --fix-lint` |

## Legacy script mapping

For reference, the following table shows the most common historical scripts and
their new equivalents.  Any script not listed here has been retired because the
underlying code fixes are complete or the behaviour is superseded by the unified
CLI.

| Legacy script | Replacement |
| --- | --- |
| `scripts/run_fast_tests.py` | `python scripts/dev.py test --profile quick` |
| `scripts/run_ci_tests.py` | `python scripts/dev.py test --profile ci` |
| `scripts/test_runner.py` | `python scripts/dev.py test --profile full` |
| `scripts/run_benchmarks.py` | `python scripts/dev.py benchmark` |
| `scripts/validate_config.py` & `scripts/validate_docs_links.py` | `python scripts/dev.py validate --check-docs` |
| `scripts/start-services.sh` | `python scripts/dev.py services start` |
| `scripts/start-monitoring.sh` / `scripts/stop-monitoring.sh` | `python scripts/dev.py services <start|stop> --stack monitoring` |

All other Python automation utilities have been removed to keep the repository
lean.  If you need a previous one-off migration tool, refer to the Git history.
