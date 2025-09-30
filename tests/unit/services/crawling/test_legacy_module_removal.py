"""Ensure deprecated Crawl4AI modules are fully removed."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "src.services.crawling.crawl4ai_provider",
        "src.services.crawling.crawl4ai_utils",
        "src.services.crawling.extractors",
    ],
)
def test_legacy_modules_absent(module_name: str) -> None:
    """Importing deprecated modules should fail to prevent stale dependencies."""

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
