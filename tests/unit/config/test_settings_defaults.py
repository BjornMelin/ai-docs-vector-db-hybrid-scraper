"""Tests for configuration defaults and precedence resolution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, ValidationError
from pydantic_settings import SettingsError
from yaml import safe_load, safe_load_all

from src.config.loader import (
    Settings,
    get_settings,
    load_settings_from_file,
    validate_settings_payload,
)
from src.config.models import (
    EmbeddingConfig,
    EmbeddingProvider,
    Environment,
    FastEmbedConfig,
    OpenAIConfig,
    SearchStrategy,
)
from src.config.template_assets import load_builtin_template_assets


# pylint: disable=no-member  # Dynamic Pydantic models expose attributes at runtime during tests.


@pytest.fixture(autouse=True)
def clear_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure cached settings instances are cleared between tests."""
    monkeypatch.delenv("AI_DOCS_CACHE__TTL_SEARCH_RESULTS", raising=False)
    monkeypatch.delenv("AI_DOCS_EMBEDDING_PROVIDER", raising=False)
    monkeypatch.delenv("AI_DOCS_OPENAI__API_KEY", raising=False)
    monkeypatch.delenv("AI_DOCS_MODE", raising=False)
    monkeypatch.delenv("AI_DOCS_QDRANT__URL", raising=False)
    monkeypatch.setattr("src.config.loader._ACTIVE_SETTINGS", None)


def _write_env_file(directory: Path, payload: str | None) -> None:
    """Create a .env file with the provided payload when requested."""
    if payload is None:
        return
    (directory / ".env").write_text(payload, encoding="utf-8")


def _model_type(annotation: Any) -> type[BaseModel] | None:
    """Return a nested Pydantic model type when the annotation defines one."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _assert_supported_setting_key(key: str) -> None:
    """Assert that an application variable maps to the canonical model tree."""
    assert key.startswith("AI_DOCS_") and not key.startswith("AI_DOCS__")
    parts = key.removeprefix("AI_DOCS_").lower().split("__")
    model: type[BaseModel] = Settings
    for index, part in enumerate(parts):
        field = model.model_fields.get(part)
        assert field is not None, f"Unsupported application setting: {key}"
        if index < len(parts) - 1:
            nested_model = _model_type(field.annotation)
            assert nested_model is not None, f"Non-nested setting path: {key}"
            model = nested_model


@dataclass(frozen=True)
class TtlCase:
    """Test case definition for TTL precedence scenarios."""

    env_value: str | None
    env_file_value: str | None
    override_value: int | None
    expected: int
    activated_value: int | None = None


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(TtlCase("7200", None, None, 7200), id="env-only"),
        pytest.param(
            TtlCase(None, "AI_DOCS_CACHE__TTL_SEARCH_RESULTS=8100\n", None, 8100),
            id="env-file",
        ),
        pytest.param(TtlCase(None, None, 9000, 9000), id="override-only"),
        pytest.param(TtlCase(None, None, None, 3600), id="defaults"),
        pytest.param(
            TtlCase(
                "600",
                "AI_DOCS_CACHE__TTL_SEARCH_RESULTS=500\n",
                700,
                700,
                400,
            ),
            id="override-wins",
        ),
        pytest.param(
            TtlCase("600", "AI_DOCS_CACHE__TTL_SEARCH_RESULTS=500\n", None, 600),
            id="env-beats-file",
        ),
        pytest.param(
            TtlCase(None, "AI_DOCS_CACHE__TTL_SEARCH_RESULTS=500\n", 700, 700),
            id="override-beats-file",
        ),
        pytest.param(TtlCase(None, None, None, 400, 400), id="activated-file"),
        pytest.param(
            TtlCase(
                None,
                "AI_DOCS_CACHE__TTL_SEARCH_RESULTS=500\n",
                None,
                500,
                400,
            ),
            id="dotenv-beats-activated",
        ),
    ],
)
def test_cache_ttl_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, case: TtlCase
) -> None:
    """Cache TTL values should resolve according to source precedence."""
    monkeypatch.chdir(tmp_path)
    if case.env_value is not None:
        monkeypatch.setenv("AI_DOCS_CACHE__TTL_SEARCH_RESULTS", case.env_value)
    _write_env_file(tmp_path, case.env_file_value)
    if case.activated_value is not None:
        (tmp_path / "config.json").write_text(
            json.dumps({"cache": {"ttl_search_results": case.activated_value}}),
            encoding="utf-8",
        )
    overrides: dict[str, Any] = {}
    if case.override_value is not None:
        overrides["cache"] = cast(Any, {"ttl_search_results": case.override_value})

    settings = Settings(**overrides)

    assert settings.cache.ttl_search_results == case.expected


def test_activated_config_is_the_cached_settings_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The activated profile should flow through the canonical cached instance."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"app_name": "Activated application"}),
        encoding="utf-8",
    )

    settings = get_settings()

    assert settings.app_name == "Activated application"
    assert get_settings() is settings


def test_activated_config_rejects_unknown_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Activated profile files retain strict canonical-field validation."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"removed_setting": True}),
        encoding="utf-8",
    )

    with pytest.raises(SettingsError, match="removed_setting"):
        Settings()


def test_payload_validation_uses_model_defaults_not_activated_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Template validation should not inherit an unrelated activated profile."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps({"app_name": "Previously activated"}),
        encoding="utf-8",
    )

    is_valid, errors, settings = validate_settings_payload({})

    assert is_valid is True
    assert not errors
    assert settings is not None
    assert settings.app_name == "AI Documentation Vector DB"


def test_cache_ttl_negative_values_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Negative TTL values should fail validation with a ``ValidationError``."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError):
        Settings(cache=cast(Any, {"ttl_search_results": -1}))


def test_chunk_overlap_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Chunk overlap exceeding the chunk size should raise ``ValidationError``."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            environment=Environment.TESTING,
            chunking=cast(Any, {"chunk_size": 100, "chunk_overlap": 200}),
        )

    assert "chunk_overlap" in str(exc_info.value)


def test_openai_provider_requires_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """OpenAI embedding provider should require an API key outside test mode."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError) as exc_info:
        Settings(embedding_provider=EmbeddingProvider.OPENAI)

    assert "OpenAI API key required" in str(exc_info.value)


def test_testing_environment_skips_provider_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test environment should skip API key validation for convenience."""
    monkeypatch.chdir(tmp_path)
    settings = Settings(
        environment=Environment.TESTING, embedding_provider=EmbeddingProvider.OPENAI
    )

    assert settings.embedding_provider is EmbeddingProvider.OPENAI


def test_env_file_invalid_value_raises_validation_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Malformed numeric values in .env should raise ``ValidationError``."""
    monkeypatch.chdir(tmp_path)
    _write_env_file(tmp_path, "AI_DOCS_CACHE__TTL_SEARCH_RESULTS=not-a-number\n")

    with pytest.raises(ValidationError):
        Settings()


def test_programmatic_overrides_apply_when_no_other_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Programmatic overrides should update nested configuration values."""
    monkeypatch.chdir(tmp_path)
    settings = Settings(cache=cast(Any, {"ttl_search_results": 4321}))

    assert settings.cache.ttl_search_results == 4321


def test_settings_payload_rejects_unknown_fields() -> None:
    """Payload validation should reject keys outside the current schema."""
    payload = {"qdrant": {"unknown_option": True}}
    is_valid, errors, settings = validate_settings_payload(payload)

    assert is_valid is False
    assert errors
    assert settings is None


def test_file_loader_rejects_unknown_nested_fields(tmp_path: Path) -> None:
    """Configuration files should reject keys outside the canonical schema."""
    config_path = tmp_path / "settings.json"
    config_path.write_text(
        json.dumps({"qdrant": {"unknown_option": True}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"qdrant\.unknown_option"):
        load_settings_from_file(config_path)


def test_env_example_contains_only_supported_settings() -> None:
    """Every application variable in `.env.example` should map to Settings."""
    env_example = Path(__file__).parents[3] / ".env.example"
    keys = [
        line.partition("=")[0].strip()
        for line in env_example.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    for key in keys:
        _assert_supported_setting_key(key)


def test_active_docs_contain_only_supported_application_settings() -> None:
    """README and active docs should use only canonical application variables."""
    repository_root = Path(__file__).parents[3]
    docs_root = repository_root / "docs"
    paths = [
        repository_root / "README.md",
        repository_root / "config" / "templates" / "README.md",
        *(
            path
            for path in docs_root.rglob("*.md")
            if path.relative_to(docs_root).parts[0] != "security"
        ),
    ]
    setting_pattern = re.compile(r"\bAI_DOCS_[A-Z][A-Z0-9_]*\b")

    for path in paths:
        for key in setting_pattern.findall(path.read_text(encoding="utf-8")):
            if not key.endswith("_"):
                _assert_supported_setting_key(key)


def test_compose_uses_supported_settings_and_profiles() -> None:
    """Compose should use real profiles and application environment fields."""
    compose_path = Path(__file__).parents[3] / "docker-compose.yml"
    compose = safe_load(compose_path.read_text(encoding="utf-8"))
    services = cast(dict[str, dict[str, Any]], compose["services"])

    profiles = {
        profile
        for service in services.values()
        for profile in cast(list[str], service.get("profiles", []))
    }
    assert profiles == {"simple", "enterprise"}

    app_environment = cast(list[str], services["app"]["environment"])
    for entry in app_environment:
        key = entry.partition("=")[0]
        if key.startswith("AI_DOCS_"):
            _assert_supported_setting_key(key)

    qdrant_healthcheck = cast(dict[str, Any], services["qdrant"]["healthcheck"])
    command = " ".join(cast(list[str], qdrant_healthcheck["test"]))
    assert "/dev/tcp/127.0.0.1/6333" in command
    assert services["qdrant"]["image"] == "qdrant/qdrant:v1.16.2"


def test_embedding_configuration_has_single_field_owners() -> None:
    """Provider models should own models and shared config should own behavior."""
    assert set(EmbeddingConfig.model_fields) == {"retrieval_mode"}
    assert {"dense_model", "sparse_model"} <= set(FastEmbedConfig.model_fields)
    assert "model" in OpenAIConfig.model_fields
    assert (
        EmbeddingConfig(retrieval_mode=SearchStrategy.HYBRID).retrieval_mode
        is SearchStrategy.HYBRID
    )


@pytest.mark.parametrize(
    ("profile_name", "expected_mode"),
    [
        ("development", SearchStrategy.HYBRID),
        ("production", SearchStrategy.HYBRID),
        ("personal-use", SearchStrategy.HYBRID),
        ("local-only", SearchStrategy.DENSE),
        ("testing", SearchStrategy.DENSE),
    ],
)
def test_profiles_place_retrieval_mode_under_embedding(
    profile_name: str,
    expected_mode: SearchStrategy,
) -> None:
    """Every opinionated profile should apply its declared retrieval mode."""
    _, profiles = load_builtin_template_assets()
    overrides = cast(dict[str, Any], profiles[profile_name]["overrides"])

    assert overrides["embedding"] == {"retrieval_mode": expected_mode.value}
    assert Settings.model_validate(overrides).embedding.retrieval_mode is expected_mode


def test_qdrant_kubernetes_manifest_matches_runtime_contract() -> None:
    """Kubernetes should use the pinned server and supported probe endpoints."""
    manifest_path = Path(__file__).parents[3] / "k8s" / "qdrant-statefulset.yaml"
    manifests = list(safe_load_all(manifest_path.read_text(encoding="utf-8")))
    stateful_set = next(item for item in manifests if item["kind"] == "StatefulSet")
    container = stateful_set["spec"]["template"]["spec"]["containers"][0]

    assert container["image"] == "qdrant/qdrant:v1.16.2"
    assert container["livenessProbe"]["httpGet"]["path"] == "/livez"
    assert container["readinessProbe"]["httpGet"]["path"] == "/readyz"


def test_application_kubernetes_manifest_matches_runtime_contract() -> None:
    """Kubernetes should probe public application endpoints and enable its cache."""
    repository_root = Path(__file__).parents[3]
    manifest_path = repository_root / "k8s" / "app-deployment.yaml"
    manifests = list(safe_load_all(manifest_path.read_text(encoding="utf-8")))
    deployment = next(item for item in manifests if item["kind"] == "Deployment")
    container = deployment["spec"]["template"]["spec"]["containers"][0]

    assert container["livenessProbe"]["httpGet"]["path"] == "/"
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"

    kustomization = safe_load(
        (repository_root / "k8s" / "kustomization.yaml").read_text(encoding="utf-8")
    )
    literals = kustomization["configMapGenerator"][0]["literals"]
    assert "AI_DOCS_CACHE__ENABLE_DRAGONFLY_CACHE=true" in literals


def test_runtime_only_ci_sync_excludes_default_dev_group() -> None:
    """Runtime-only workflow setup should not reinstall the default dev group."""
    repository_root = Path(__file__).parents[3]
    setup_action = safe_load(
        (
            repository_root / ".github" / "actions" / "setup-environment" / "action.yml"
        ).read_text(encoding="utf-8")
    )
    install_script = next(
        step["run"]
        for step in setup_action["runs"]["steps"]
        if step["name"] == "Install dependencies"
    )
    assert "uv sync --frozen --no-dev" in install_script

    docs_workflow = safe_load(
        (repository_root / ".github" / "workflows" / "docs.yml").read_text(
            encoding="utf-8"
        )
    )
    documentation_steps = docs_workflow["jobs"]["docs"]["steps"]
    install_docs = next(
        step["run"]
        for step in documentation_steps
        if step["name"] == "Install documentation dependencies"
    )
    assert install_docs == "uv sync --frozen --no-dev --extra docs"


def test_env_prefix_and_nested_delimiter(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Canonical top-level and nested variables should apply without extra prefix."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI_DOCS_MODE", "contract-test")
    monkeypatch.setenv("AI_DOCS_QDRANT__URL", "http://qdrant.test:6333")

    settings = Settings()

    assert settings.mode == "contract-test"
    assert settings.qdrant.url == "http://qdrant.test:6333"
