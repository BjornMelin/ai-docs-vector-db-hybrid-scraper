"""Tests for the configuration validation helper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts.ci import validate_config


def test_validate_json_files_reports_errors(tmp_path: Path) -> None:
    """The validator should surface JSON parsing failures."""
    config_root = tmp_path / "config"
    config_root.mkdir()
    (config_root / "valid.json").write_text("{}\n", encoding="utf-8")
    (config_root / "invalid.json").write_text("{", encoding="utf-8")

    summary = validate_config.validate_json_files(config_root)

    assert summary.checked == 2
    assert any("Invalid JSON" in message for message in summary.errors)


def test_validate_yaml_files_reports_errors(tmp_path: Path) -> None:
    """The validator should flag YAML parsing failures."""
    config_root = tmp_path / "config"
    config_root.mkdir()
    (config_root / "valid.yaml").write_text("key: value\n", encoding="utf-8")
    (config_root / "invalid.yaml").write_text(": bad\n", encoding="utf-8")

    summary = validate_config.validate_yaml_files(config_root)

    assert summary.checked == 2
    assert any("Invalid YAML" in message for message in summary.errors)


def test_validate_templates_handles_environment_mismatch(tmp_path: Path) -> None:
    """Environment enforcement should catch template mismatches."""
    templates_dir = tmp_path / "config" / "templates"
    templates_dir.mkdir(parents=True)
    (templates_dir / "base.json").write_text(
        json.dumps(
            {
                "environment": "development",
                "cache": {},
                "qdrant": {},
                "performance": {},
            }
        ),
        encoding="utf-8",
    )
    (templates_dir / "profiles.json").write_text(
        json.dumps(
            {
                "development": {
                    "overrides": {
                        "environment": "production",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (templates_dir / "development.json").write_text(
        json.dumps(
            {"environment": "production", "cache": {}, "qdrant": {}, "performance": {}}
        ),
        encoding="utf-8",
    )

    summary = validate_config.validate_templates(
        templates_dir,
        environment="development",
    )

    assert summary.checked == 1
    assert summary.errors
    assert "development" in summary.errors[0]


def test_validate_templates_inherits_settings_defaults(tmp_path: Path) -> None:
    """Partial templates should inherit omitted sections from Settings."""
    templates_dir = tmp_path / "config" / "templates"
    templates_dir.mkdir(parents=True)
    (templates_dir / "base.json").write_text(
        json.dumps({"environment": "development"}), encoding="utf-8"
    )
    (templates_dir / "profiles.json").write_text(
        json.dumps({"minimal": {"overrides": {}}}), encoding="utf-8"
    )

    summary = validate_config.validate_templates(templates_dir, environment=None)

    assert summary.checked == 1
    assert not summary.errors


def test_validate_templates_uses_packaged_assets_by_default() -> None:
    """The CI validator should inspect the same assets consumed by the wizard."""
    summary = validate_config.validate_templates(None, environment="development")

    assert summary.checked == 1
    assert not summary.errors


def test_validate_templates_rejects_unknown_settings(tmp_path: Path) -> None:
    """Template validation should reject removed fields instead of ignoring them."""
    templates_dir = tmp_path / "config" / "templates"
    templates_dir.mkdir(parents=True)
    (templates_dir / "base.json").write_text("{}\n", encoding="utf-8")
    (templates_dir / "profiles.json").write_text(
        json.dumps(
            {"legacy": {"overrides": {"fastembed": {"retrieval_mode": "hybrid"}}}}
        ),
        encoding="utf-8",
    )

    summary = validate_config.validate_templates(templates_dir, environment=None)

    assert summary.checked == 1
    assert any("fastembed.retrieval_mode" in error for error in summary.errors)


def test_main_handles_missing_config_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI should fail fast when the provided config root is absent."""
    missing_root = tmp_path / "does-not-exist"
    exit_code = validate_config.main(
        [
            "--config-root",
            str(missing_root),
            "--templates-dir",
            str(missing_root / "templates"),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Configuration root not found" in captured.err
