"""Deployment manifest contracts."""

from pathlib import Path
from typing import Any, cast

from yaml import safe_load, safe_load_all


PROJECT_ROOT = Path(__file__).parents[3]
DRAGONFLY_IMAGE = "docker.dragonflydb.io/dragonflydb/dragonfly:v1.39.0"


def test_kustomize_applies_the_multi_document_patch_bundle_once() -> None:
    """A multi-document patch cannot be reused with per-entry target filters."""
    kustomization = safe_load(
        (PROJECT_ROOT / "k8s/kustomization.yaml").read_text(encoding="utf-8")
    )
    patches = list(
        safe_load_all(
            (PROJECT_ROOT / "k8s/patches/production-resources.yaml").read_text(
                encoding="utf-8"
            )
        )
    )

    assert kustomization["patches"] == [{"path": "patches/production-resources.yaml"}]
    assert {(patch["kind"], patch["metadata"]["name"]) for patch in patches} == {
        ("Deployment", "ai-docs-app"),
        ("Deployment", "dragonfly"),
        ("StatefulSet", "qdrant"),
    }

    resources = set(kustomization["resources"])
    assert "worker-deployment.yaml" not in resources

    common_labels = kustomization["labels"][0]
    assert "app.kubernetes.io/name" not in common_labels["pairs"]
    assert common_labels.get("includeSelectors", False) is False

    qdrant_patch = next(
        patch for patch in patches if patch["metadata"]["name"] == "qdrant"
    )
    claim = qdrant_patch["spec"]["volumeClaimTemplates"][0]["spec"]
    assert claim["accessModes"] == ["ReadWriteOnce"]
    assert claim["storageClassName"] == "gp2"


def test_dragonfly_manifests_use_only_supported_command_flags() -> None:
    """Dragonfly should use documented flags without inert environment aliases."""
    compose_args = ["--logtostderr", "--cache_mode"]
    kubernetes_args = [*compose_args, "--maxmemory=3gb"]

    compose = safe_load(
        (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    compose_service = cast(dict[str, Any], compose["services"]["dragonfly"])

    manifests = list(
        safe_load_all(
            (PROJECT_ROOT / "k8s/dragonfly-deployment.yaml").read_text(encoding="utf-8")
        )
    )
    deployment = next(item for item in manifests if item["kind"] == "Deployment")
    container = cast(
        dict[str, Any], deployment["spec"]["template"]["spec"]["containers"][0]
    )
    production_patches = list(
        safe_load_all(
            (PROJECT_ROOT / "k8s/patches/production-resources.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    dragonfly_patch = next(
        item for item in production_patches if item["metadata"]["name"] == "dragonfly"
    )
    patched_container = cast(
        dict[str, Any],
        dragonfly_patch["spec"]["template"]["spec"]["containers"][0],
    )

    assert compose_service["command"] == compose_args
    assert compose_service["image"] == DRAGONFLY_IMAGE
    assert "environment" not in compose_service
    assert container["args"] == kubernetes_args
    assert container["image"] == DRAGONFLY_IMAGE
    assert "env" not in container
    assert "env" not in patched_container


def test_monitoring_images_use_stable_version_tags() -> None:
    """Compose should never resolve monitoring services through floating tags."""
    compose = safe_load(
        (PROJECT_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    services = cast(dict[str, dict[str, Any]], compose["services"])

    assert services["prometheus"]["image"] == "prom/prometheus:v3.12.0"
    assert services["grafana"]["image"] == "grafana/grafana:13.0.2"


def test_kubernetes_guide_commands_are_repository_root_relative() -> None:
    """Every Kustomize command should work from the documented repository root."""
    guide = (PROJECT_ROOT / "k8s/README.md").read_text(encoding="utf-8")

    assert "kubectl apply -k ." not in guide
    assert "kustomize build ." not in guide
    assert "kubectl delete -k ." not in guide
    assert "kubectl apply -k k8s" in guide
    assert "kustomize build k8s" in guide
    assert "kubectl delete -k k8s" in guide
