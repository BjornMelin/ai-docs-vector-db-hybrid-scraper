"""Regression tests for the unified development CLI wrappers."""

from pathlib import Path
from subprocess import CompletedProcess

import pytest
from click.testing import CliRunner

from src.cli import unified


@pytest.mark.parametrize(
    "arguments",
    [
        ["test"],
        ["docs"],
        ["benchmark"],
        ["eval"],
        ["quality"],
        ["validate"],
    ],
)
def test_subprocess_commands_propagate_failure(
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
) -> None:
    """Subprocess-backed commands should preserve their non-zero exit status."""
    monkeypatch.setattr(
        unified,
        "_run_command",
        lambda command, **kwargs: CompletedProcess(command, 9),
    )

    result = CliRunner().invoke(unified.cli, arguments)

    assert result.exit_code == 9


def test_services_propagates_dev_script_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed service command should produce the same non-zero exit status."""
    calls: list[list[str]] = []

    def run_command(command: list[str], **_: object) -> CompletedProcess[str]:
        calls.append(command)
        return CompletedProcess(command, 7)

    monkeypatch.setattr(unified, "_run_command", run_command)

    result = CliRunner().invoke(
        unified.cli,
        ["services", "--action", "status", "--stack", "enterprise"],
    )

    assert result.exit_code == 7
    assert calls[0][-4:] == ["services", "status", "--stack", "enterprise"]


def test_setup_creates_canonical_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Setup should copy `.env.example` to the file loaded by Settings."""
    repo_root = tmp_path
    env_example = repo_root / ".env.example"
    env_example.write_text("AI_DOCS_MODE=contract-test\n", encoding="utf-8")
    monkeypatch.setattr(unified, "REPO_ROOT", repo_root)
    monkeypatch.setattr(
        unified,
        "_run_command",
        lambda command, **kwargs: CompletedProcess(command, 0),
    )

    result = CliRunner().invoke(unified.cli, ["setup"])

    assert result.exit_code == 0
    assert (repo_root / ".env").read_text(encoding="utf-8") == (
        "AI_DOCS_MODE=contract-test\n"
    )


def test_setup_propagates_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Setup should preserve the validation helper's exit status."""
    (tmp_path / ".env.example").write_text(
        "AI_DOCS_MODE=contract-test\n", encoding="utf-8"
    )
    monkeypatch.setattr(unified, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        unified,
        "_run_command",
        lambda command, **kwargs: CompletedProcess(command, 9),
    )

    result = CliRunner().invoke(unified.cli, ["setup"])

    assert result.exit_code == 9
