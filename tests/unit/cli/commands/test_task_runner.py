"""Tests for the task runner Click commands."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from src.cli.commands import task_runner


@patch("src.cli.commands.task_runner.uvicorn.run")
def test_dev_command_sets_mode_and_invokes_uvicorn(mock_uvicorn: MagicMock) -> None:
    """Ensure the dev command configures the environment and calls uvicorn."""

    runner = CliRunner()
    result = runner.invoke(
        task_runner.dev,
        [
            "--mode",
            "enterprise",
            "--host",
            "127.0.0.1",
            "--port",
            "9001",
            "--no-reload",
        ],
    )

    assert result.exit_code == 0
    assert os.environ.get("AI_DOCS__MODE") == "enterprise"
    mock_uvicorn.assert_called_once_with(
        task_runner.APP_IMPORT_PATH,
        reload=False,
        host="127.0.0.1",
        port=9001,
    )


@patch("src.cli.commands.task_runner.dev_script.main", return_value=0)
def test_test_command_invokes_dev_script(mock_main: MagicMock) -> None:
    """Verify that test command forwards arguments to scripts.dev."""

    runner = CliRunner()
    result = runner.invoke(
        task_runner.test,
        [
            "--profile",
            "unit",
            "--coverage",
            "--verbose",
            "--workers",
            "3",
            "--",
            "--maxfail=1",
        ],
    )

    assert result.exit_code == 0
    mock_main.assert_called_once_with(
        [
            "test",
            "--profile",
            "unit",
            "--coverage",
            "--verbose",
            "--workers",
            "3",
            "--",
            "--maxfail=1",
        ]
    )


@patch("src.cli.commands.task_runner.dev_script.main", return_value=0)
def test_quality_command_respects_flags(mock_main: MagicMock) -> None:
    """Ensure quality command toggles optional flags."""

    runner = CliRunner()
    result = runner.invoke(task_runner.quality, ["--skip-format", "--no-fix-lint"])

    assert result.exit_code == 0
    mock_main.assert_called_once_with(["quality", "--skip-format"])


@patch("src.cli.commands.task_runner.subprocess.run")
def test_docs_command_invokes_mkdocs(mock_run: MagicMock) -> None:
    """Ensure docs command shells out to mkdocs serve."""

    mock_run.return_value = MagicMock(returncode=0)
    runner = CliRunner()
    result = runner.invoke(task_runner.docs, ["--host", "127.0.0.1", "--port", "9000"])

    assert result.exit_code == 0
    mock_run.assert_called_once_with(
        ["mkdocs", "serve", "--host", "127.0.0.1", "--port", "9000"],
        check=False,
        env=None,
        shell=False,
    )


@patch("src.cli.commands.task_runner.dev_script.main", return_value=0)
def test_services_command_invokes_dev_script(mock_main: MagicMock) -> None:
    """Ensure services command forwards flags to scripts.dev."""

    runner = CliRunner()
    result = runner.invoke(
        task_runner.services,
        ["--action", "status", "--stack", "monitoring", "--skip-health-check"],
    )

    assert result.exit_code == 0
    mock_main.assert_called_once_with(
        ["services", "status", "--stack", "monitoring", "--skip-health-check"]
    )


@patch("src.cli.commands.task_runner.dev_script.main", return_value=0)
def test_benchmark_command_invokes_dev_script(mock_main: MagicMock) -> None:
    """Ensure benchmark command forwards to scripts.dev with the selected suite."""

    runner = CliRunner()
    result = runner.invoke(task_runner.benchmark, ["--profile", "integration"])

    assert result.exit_code == 0
    mock_main.assert_called_once_with(["benchmark", "--suite", "integration"])


@patch("src.cli.commands.task_runner.dev_script.main", return_value=0)
def test_eval_command_supports_optional_arguments(mock_main: MagicMock) -> None:
    """Ensure eval command forwards the full option set."""

    runner = CliRunner()
    result = runner.invoke(
        task_runner.run_eval,
        [
            "--dataset",
            "dataset.jsonl",
            "--output",
            "report.json",
            "--limit",
            "10",
            "--namespace",
            "custom",
            "--enable-ragas",
            "--ragas-model",
            "gpt",
            "--ragas-embedding",
            "text-embedding",
            "--ragas-max-samples",
            "25",
            "--metrics-allowlist",
            "latency",
            "--metrics-allowlist",
            "tokens",
        ],
    )

    assert result.exit_code == 0
    mock_main.assert_called_once_with(
        [
            "eval",
            "--dataset",
            "dataset.jsonl",
            "--limit",
            "10",
            "--namespace",
            "custom",
            "--output",
            "report.json",
            "--enable-ragas",
            "--ragas-model",
            "gpt",
            "--ragas-embedding",
            "text-embedding",
            "--ragas-max-samples",
            "25",
            "--metrics-allowlist",
            "latency",
            "tokens",
        ]
    )


@patch("src.cli.commands.task_runner.dev_script.main", return_value=0)
def test_validate_command_invokes_dev_script(mock_main: MagicMock) -> None:
    """Ensure validate command forwards the documentation flag."""

    runner = CliRunner()
    result = runner.invoke(task_runner.validate)

    assert result.exit_code == 0
    mock_main.assert_called_once_with(["validate", "--check-docs"])


@patch("src.cli.commands.task_runner.dev_script.main", return_value=1)
def test_dev_script_failure_surfaces_as_click_exception(mock_main: MagicMock) -> None:
    """Commands should raise a ClickException when scripts.dev fails."""

    runner = CliRunner()
    result = runner.invoke(task_runner.benchmark, [])

    assert result.exit_code != 0
    assert "failed with exit code" in result.output
    mock_main.assert_called_once_with(["benchmark", "--suite", "standard"])
