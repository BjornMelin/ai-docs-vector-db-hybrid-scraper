"""Delivery workflow contracts."""

from pathlib import Path
from typing import Any, cast

from yaml import safe_load


PROJECT_ROOT = Path(__file__).parents[3]
WORKFLOWS = PROJECT_ROOT / ".github/workflows"


def _workflow(name: str) -> dict[str, Any]:
    """Load a GitHub Actions workflow with PyYAML's YAML 1.1 key handling."""
    payload = cast(
        dict[str | bool, Any],
        safe_load((WORKFLOWS / name).read_text(encoding="utf-8")),
    )
    if True in payload:
        payload["on"] = payload.pop(True)
    return cast(dict[str, Any], payload)


def test_ci_builds_the_root_container_for_precise_inputs() -> None:
    """Container-affecting files should trigger a non-pushing production build."""
    workflow = _workflow("ci.yml")
    filters = safe_load(
        next(
            step["with"]["filters"]
            for step in workflow["jobs"]["changes"]["steps"]
            if step.get("id") == "filter"
        )
    )

    assert set(filters["container"]) == {
        ".dockerignore",
        ".github/workflows/ci.yml",
        "Dockerfile",
        "README.md",
        "config/**",
        "pyproject.toml",
        "src/**",
        "uv.lock",
    }

    container_job = workflow["jobs"]["container"]
    build = next(
        step
        for step in container_job["steps"]
        if step["name"] == "Build production image"
    )
    assert build["uses"].startswith("docker/build-push-action@")
    assert build["uses"].split("@", maxsplit=1)[1].isalnum()
    assert build["with"]["context"] == "."
    assert build["with"]["file"] == "Dockerfile"
    assert build["with"]["push"] is False


def test_kubernetes_workflow_renders_and_schema_validates_synthetic_input() -> None:
    """Kubernetes changes should render and validate without repository secrets."""
    workflow = _workflow("config-deployment.yml")
    events = workflow["on"]
    assert "k8s/**" in events["push"]["paths"]
    assert "k8s/**" in events["pull_request"]["paths"]

    filters = safe_load(workflow["env"]["CONFIG_PATH_FILTER"])
    assert "k8s/**" in filters["config"]

    validate_job = workflow["jobs"]["validate-config"]
    render = next(
        step
        for step in validate_job["steps"]
        if step["name"] == "Render and validate Kubernetes manifests"
    )
    script = cast(str, render["run"])
    assert "ci-placeholder" in script
    assert "kustomize/kustomize/v5@v5.7.1 build k8s" in script
    assert "kubeconform/cmd/kubeconform@v0.7.0" in script
    assert "${{ secrets." not in script


def test_docs_workflow_tracks_locked_dependencies_without_ad_hoc_pytest() -> None:
    """Docs dependency changes should run the lock-backed MkDocs build."""
    workflow = _workflow("docs.yml")
    events = workflow["on"]
    for event in ("push", "pull_request"):
        paths = set(events[event]["paths"])
        assert {"pyproject.toml", "uv.lock"} <= paths

    steps = workflow["jobs"]["docs"]["steps"]
    commands = "\n".join(str(step.get("run", "")) for step in steps)
    assert "uv sync --frozen --no-dev --extra docs" in commands
    assert "pytest" not in commands
