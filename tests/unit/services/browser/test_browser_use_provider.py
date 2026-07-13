"""Tests for the Browser-use provider adapter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.config.browser import BrowserUseSettings
from src.services.browser.errors import BrowserProviderError
from src.services.browser.models import ProviderKind, ScrapeRequest
from src.services.browser.providers.base import ProviderContext
from src.services.browser.providers.browser_use import BrowserUseProvider


agent_views = pytest.importorskip("browser_use.agent.views")
browser_views = pytest.importorskip("browser_use.browser.views")
ActionResult = agent_views.ActionResult
AgentHistory = agent_views.AgentHistory
AgentHistoryList = agent_views.AgentHistoryList
BrowserStateHistory = browser_views.BrowserStateHistory


@pytest.mark.asyncio
async def test_browser_use_uses_current_browser_session_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter should pass launch settings directly to BrowserSession."""
    browser_kwargs: dict[str, Any] = {}
    llm_kwargs: dict[str, Any] = {}

    class FakeBrowser:
        def __init__(self, **kwargs: Any) -> None:
            browser_kwargs.update(kwargs)

        async def kill(self) -> None:
            return None

    class FakeAgent:
        def __init__(self, **_kwargs: Any) -> None:
            return None

        async def run(self) -> Any:
            state = BrowserStateHistory(
                url="https://example.com",
                title="Example",
                tabs=[],
                interacted_element=[],
            )
            action = ActionResult(
                extracted_content="browser result",
                is_done=True,
                success=True,
            )
            return AgentHistoryList(
                history=[AgentHistory(model_output=None, result=[action], state=state)]
            )

    def fake_llm(**kwargs: Any) -> object:
        llm_kwargs.update(kwargs)
        return object()

    def fake_import(name: str) -> object:
        if name == "browser_use":
            return SimpleNamespace(
                Agent=FakeAgent,
                Browser=FakeBrowser,
                ChatOpenAI=fake_llm,
            )
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "src.services.browser.providers.browser_use.import_module",
        fake_import,
    )
    provider = BrowserUseProvider(
        ProviderContext(ProviderKind.BROWSER_USE),
        BrowserUseSettings(headless=True),
        openai_api_key="sk-test",
    )

    await provider.initialize()
    result = await provider.scrape(ScrapeRequest(url="https://example.com"))

    assert browser_kwargs == {"headless": True}
    assert llm_kwargs == {
        "api_key": "sk-test",
        "model": "gpt-4o-mini",
        "temperature": 0.0,
    }
    assert result.content == "browser result"
    assert result.title == "Example"
    assert result.success is True


@pytest.mark.asyncio
async def test_browser_use_native_llm_satisfies_agent_contract() -> None:
    """The pinned browser-use Agent should accept its native OpenAI model."""
    provider = BrowserUseProvider(
        ProviderContext(ProviderKind.BROWSER_USE),
        BrowserUseSettings(),
        openai_api_key="sk-test",
    )

    dependencies = provider._load_dependencies()  # pylint: disable=protected-access
    browser = dependencies.browser_cls(headless=True)
    agent = dependencies.agent_cls(
        task="Open https://example.com",
        llm=dependencies.llm,
        browser=browser,
    )

    assert dependencies.llm.provider == "openai"
    assert agent.llm is dependencies.llm
    await browser.kill()


@pytest.mark.asyncio
async def test_browser_use_normalizes_llm_configuration_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider-specific credential failures should remain optional to the router."""

    class MissingCredentialError(Exception):
        """Stand in for an SDK-specific credential exception."""

    class FailingLLM:
        def __init__(self, **_kwargs: Any) -> None:
            raise MissingCredentialError("missing provider credential")

    def fake_import(name: str) -> object:
        if name == "browser_use":
            return SimpleNamespace(
                Agent=object,
                Browser=object,
                ChatOpenAI=FailingLLM,
            )
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(
        "src.services.browser.providers.browser_use.import_module",
        fake_import,
    )
    provider = BrowserUseProvider(
        ProviderContext(ProviderKind.BROWSER_USE),
        BrowserUseSettings(),
    )

    with pytest.raises(BrowserProviderError, match="LLM is not configured"):
        await provider.initialize()
