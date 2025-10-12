"""Load and performance testing module.

Load/volume/stress suites rely on Locust and gevent. To keep unit test runs
lightweight, these suites are ignored unless ``RUN_LOAD_TESTS=1`` is present in
the environment. This keeps KISS/YAGNI for normal CI runs while retaining the
full suites for dedicated performance pipelines.
"""

import os


if os.getenv("RUN_LOAD_TESTS") != "1":  # pragma: no cover - simple configuration
    collect_ignore = [
        "base_load_test.py",
        "load_testing",
        "stress_testing",
        "spike_testing",
        "endurance_testing",
        "volume_testing",
        "scalability",
        "locust_load_runner.py",
        "runner_cli.py",
        "runner_core.py",
    ]
