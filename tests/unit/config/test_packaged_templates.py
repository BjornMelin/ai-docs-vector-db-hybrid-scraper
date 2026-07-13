"""Distribution regressions for built-in setup assets."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parents[3]


@pytest.mark.slow
def test_wheel_runs_setup_from_an_unrelated_directory(tmp_path: Path) -> None:
    """The installed wheel should own every asset needed to activate a profile."""
    dist_dir = tmp_path / "dist"
    target_dir = tmp_path / "installed"
    working_dir = tmp_path / "unrelated"
    working_dir.mkdir()

    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=PROJECT_ROOT,
        check=True,
        timeout=300,
    )
    wheel = next(dist_dir.glob("*.whl"))
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--target",
            str(target_dir),
            "--no-deps",
            str(wheel),
        ],
        check=True,
        timeout=300,
    )

    script = textwrap.dedent(
        f"""
        import os
        import sys
        from pathlib import Path

        sys.path.insert(0, {str(target_dir)!r})

        import src
        from click.testing import CliRunner
        from src.cli.commands.setup import ConfigurationWizard, setup
        from src.config.loader import Settings

        target = Path({str(target_dir)!r})
        working = Path({str(working_dir)!r})
        assert Path(src.__file__).is_relative_to(target)

        wizard = ConfigurationWizard(working / "config")
        assert "minimal" in wizard.template_manager.list_templates()
        profile = wizard.profile_manager.create_profile_config("minimal")
        assert profile.is_file()

        os.chdir(working)
        wizard.profile_manager.activate_profile("minimal")
        assert Settings().environment.value == "development"
        assert CliRunner().invoke(setup, ["--help"]).exit_code == 0
        """
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("AI_DOCS_") and key != "PYTHONPATH"
    }

    subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=working_dir,
        env=environment,
        check=True,
        timeout=300,
    )
