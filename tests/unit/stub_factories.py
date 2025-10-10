"""Utilities for registering dependency stubs used in unit tests."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Self, cast

from pydantic import BaseModel


def _ensure_module(name: str) -> types.ModuleType:
    """Return an existing module or register a new stub module."""

    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


def _set_attributes(module: types.ModuleType, attributes: dict[str, object]) -> None:
    """Assign attributes to a module if they are missing."""

    for attr_name, attr_value in attributes.items():
        if not hasattr(module, attr_name):
            setattr(module, attr_name, attr_value)


class _ChatPromptTemplate:
    """Simple stub for LangChain chat prompt templates."""

    def __init__(self, **kwargs: object) -> None:
        self.messages: list[object] = []
        self.options = dict(kwargs)

    @classmethod
    def from_messages(
        cls, messages: Iterable[object] | None = None, **kwargs: object
    ) -> _ChatPromptTemplate:
        template = cls(**kwargs)
        template.messages = list(messages or [])
        return template


def _apply_module_specs(specs: Mapping[str, Mapping[str, object]]) -> None:
    """Apply attribute specs to modules, creating the modules if missing."""

    for module_name, attributes in specs.items():
        module = _ensure_module(module_name)
        _set_attributes(module, dict(attributes))


def _ensure_namespace_packages(namespaces: Iterable[str]) -> None:
    """Ensure namespace packages expose a __path__ attribute."""

    for namespace in namespaces:
        package = _ensure_module(namespace)
        if not hasattr(package, "__path__"):
            package.__path__ = []  # type: ignore[attr-defined]


def _register_client_stubs() -> None:
    """Register stubs for client SDKs leveraged by the RAG pipeline."""

    _apply_module_specs(
        {
            "firecrawl": {
                "AsyncFirecrawlApp": type("AsyncFirecrawlApp", (), {}),
                "AsyncFirecrawl": type("AsyncFirecrawl", (), {}),
            },
            "openai": {"AsyncOpenAI": type("AsyncOpenAI", (), {})},
            "langchain_openai": {"ChatOpenAI": type("ChatOpenAI", (), {})},
            "langchain_qdrant": {
                "QdrantVectorStore": type("QdrantVectorStore", (), {}),
                "RetrievalMode": type("RetrievalMode", (), {}),
                "FastEmbedSparse": type("FastEmbedSparse", (), {}),
            },
        }
    )


def _register_langchain_retriever_stubs() -> None:
    """Install retriever-related stubs for LangChain modules."""

    langchain_module = cast(Any, _ensure_module("langchain"))
    retrievers_module = cast(Any, _ensure_module("langchain.retrievers"))
    compressor_module = cast(
        Any, _ensure_module("langchain.retrievers.document_compressors")
    )
    _set_attributes(
        compressor_module,
        {
            "DocumentCompressorPipeline": type("DocumentCompressorPipeline", (), {}),
            "EmbeddingsFilter": type("EmbeddingsFilter", (), {}),
        },
    )
    retrievers_module.document_compressors = compressor_module
    langchain_module.retrievers = retrievers_module
    _set_attributes(
        retrievers_module,
        {
            "ContextualCompressionRetriever": type(
                "ContextualCompressionRetriever", (), {}
            ),
        },
    )
    if not hasattr(langchain_module, "__path__"):
        langchain_module.__path__ = []  # type: ignore[attr-defined]


def _register_langchain_callback_stubs() -> None:
    """Install callback stubs for LangChain core modules."""

    manager_module = cast(Any, _ensure_module("langchain_core.callbacks.manager"))
    _set_attributes(
        manager_module,
        {
            "AsyncCallbackManagerForRetrieverRun": type(
                "AsyncCallbackManagerForRetrieverRun", (), {}
            ),
            "CallbackManagerForRetrieverRun": type(
                "CallbackManagerForRetrieverRun", (), {}
            ),
            "AsyncCallbackManager": type("AsyncCallbackManager", (), {}),
            "CallbackManager": type("CallbackManager", (), {}),
        },
    )

    callbacks_root = cast(Any, _ensure_module("langchain_core.callbacks"))
    _set_attributes(
        callbacks_root,
        {
            "AsyncCallbackManager": type("AsyncCallbackManager", (), {}),
            "CallbackManager": type("CallbackManager", (), {}),
            "BaseCallbackManager": type("BaseCallbackManager", (), {}),
            "Callbacks": type("Callbacks", (), {}),
            "AsyncCallbackManagerForToolRun": type(
                "AsyncCallbackManagerForToolRun", (), {}
            ),
            "CallbackManagerForToolRun": type("CallbackManagerForToolRun", (), {}),
        },
    )
    if not hasattr(callbacks_root, "__path__"):
        callbacks_root.__path__ = []  # type: ignore[attr-defined]

    callbacks_base = cast(Any, _ensure_module("langchain_core.callbacks.base"))
    _set_attributes(
        callbacks_base,
        {
            "BaseCallbackManager": type("BaseCallbackManager", (), {}),
            "AsyncCallbackHandler": type("AsyncCallbackHandler", (), {}),
            "BaseCallbackHandler": type("BaseCallbackHandler", (), {}),
        },
    )


def _register_langchain_tool_stubs() -> None:
    """Install tooling and prompt stubs for LangChain."""

    tools_module = cast(Any, _ensure_module("langchain_core.tools"))
    if not hasattr(tools_module, "__path__"):
        tools_module.__path__ = []  # type: ignore[attr-defined]
    _set_attributes(
        tools_module,
        {
            "BaseTool": type("BaseTool", (), {}),
            "InjectedToolArg": type("InjectedToolArg", (), {}),
            "StructuredTool": type("StructuredTool", (), {}),
            "Tool": type("Tool", (), {}),
            "ToolException": type("ToolException", (), {}),
        },
    )

    tool_base_module = cast(Any, _ensure_module("langchain_core.tools.base"))
    _set_attributes(
        tool_base_module,
        {
            "BaseTool": type("BaseTool", (), {}),
            "get_all_basemodel_annotations": lambda *args, **kwargs: {},
        },
    )

    prompts_module = cast(Any, _ensure_module("langchain_core.prompts"))
    _set_attributes(prompts_module, {"ChatPromptTemplate": _ChatPromptTemplate})


def _register_langchain_document_stubs() -> None:
    """Install document and embedding stubs for LangChain."""

    documents_module = cast(Any, _ensure_module("langchain_core.documents"))
    _set_attributes(documents_module, {"Document": type("Document", (), {})})

    documents_base = cast(Any, _ensure_module("langchain_core.documents.base"))
    _set_attributes(documents_base, {"Blob": type("Blob", (), {})})

    retrievers_core = cast(Any, _ensure_module("langchain_core.retrievers"))
    _set_attributes(retrievers_core, {"BaseRetriever": type("BaseRetriever", (), {})})

    embeddings_core = cast(Any, _ensure_module("langchain_core.embeddings"))
    _set_attributes(embeddings_core, {"Embeddings": type("Embeddings", (), {})})

    runnables_module = cast(Any, _ensure_module("langchain_core.runnables"))
    _set_attributes(
        runnables_module,
        {
            "RunnableConfig": type("RunnableConfig", (), {}),
        },
    )


def _register_langchain_text_splitters() -> None:
    """Install text splitter stubs for LangChain."""

    text_splitters = cast(Any, _ensure_module("langchain_text_splitters"))
    _set_attributes(
        text_splitters,
        {
            "RecursiveCharacterTextSplitter": type(
                "RecursiveCharacterTextSplitter", (), {}
            ),
        },
    )


def _register_langchain_message_stubs() -> None:
    """Install message primitives for LangChain."""

    messages_module = cast(Any, _ensure_module("langchain_core.messages"))
    _set_attributes(
        messages_module,
        {
            "AIMessage": type("AIMessage", (), {}),
            "HumanMessage": type("HumanMessage", (), {}),
        },
    )


def _register_langchain_core_stubs() -> None:
    """Register core LangChain modules required for RAG tests."""

    _register_langchain_retriever_stubs()
    _register_langchain_callback_stubs()
    _register_langchain_tool_stubs()
    _register_langchain_document_stubs()
    _register_langchain_text_splitters()
    _register_langchain_message_stubs()


def _register_langchain_community_stubs() -> None:
    """Register LangChain community modules used by the RAG pipeline."""

    community = cast(Any, _ensure_module("langchain_community"))
    community_embeddings = cast(Any, _ensure_module("langchain_community.embeddings"))
    fastembed = cast(Any, _ensure_module("langchain_community.embeddings.fastembed"))

    class _StubFastEmbedEmbeddings:
        """Minimal stub of LangChain FastEmbed embeddings."""

        max_length = 512

        def __init__(self, **_kwargs: Any) -> None:  # noqa: D401 - simple stub
            self._vector = [0.1, 0.2, 0.3]

        def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
            return [list(self._vector) for _ in texts]

        def embed_query(self, _text: str) -> list[float]:
            return list(self._vector)

    _set_attributes(
        fastembed,
        {"FastEmbedEmbeddings": _StubFastEmbedEmbeddings},
    )
    community_embeddings.fastembed = fastembed
    community_embeddings.FastEmbedEmbeddings = fastembed.FastEmbedEmbeddings
    community.embeddings = community_embeddings

    transformers = cast(
        Any, _ensure_module("langchain_community.document_transformers")
    )
    _set_attributes(
        transformers,
        {"EmbeddingsRedundantFilter": type("EmbeddingsRedundantFilter", (), {})},
    )

    if not hasattr(community, "__path__"):
        community.__path__ = []  # type: ignore[attr-defined]


def _register_crawl4ai_stubs() -> None:
    """Register Crawl4AI modules required for crawling stubs."""

    _apply_module_specs(
        {
            "crawl4ai": {
                "AsyncWebCrawler": type("AsyncWebCrawler", (), {}),
                "CacheMode": type("CacheMode", (), {"BYPASS": "bypass"}),
                "BrowserConfig": type("BrowserConfig", (), {}),
                "CrawlerRunConfig": type("CrawlerRunConfig", (), {}),
                "MemoryAdaptiveDispatcher": type("MemoryAdaptiveDispatcher", (), {}),
                "DefaultMarkdownGenerator": type("DefaultMarkdownGenerator", (), {}),
                "LinkPreviewConfig": type("LinkPreviewConfig", (), {}),
                "RunConfig": type("RunConfig", (), {"clone": lambda self, **_: self}),
                "CrawlerMonitor": type("CrawlerMonitor", (), {}),
            },
            "crawl4ai.models": {"CrawlResult": type("CrawlResult", (), {})},
            "crawl4ai.async_dispatcher": {
                "SemaphoreDispatcher": type("SemaphoreDispatcher", (), {})
            },
            "crawl4ai.deep_crawling": {
                "BestFirstCrawlingStrategy": type("BestFirstCrawlingStrategy", (), {}),
                "BFSDeepCrawlStrategy": type("BFSDeepCrawlStrategy", (), {}),
            },
            "crawl4ai.deep_crawling.filters": {
                "ContentRelevanceFilter": type("ContentRelevanceFilter", (), {}),
                "ContentTypeFilter": type("ContentTypeFilter", (), {}),
                "DomainFilter": type("DomainFilter", (), {}),
                "FilterChain": type("FilterChain", (), {}),
                "SEOFilter": type("SEOFilter", (), {}),
                "URLFilter": type("URLFilter", (), {}),
                "URLPatternFilter": type("URLPatternFilter", (), {}),
            },
            "crawl4ai.deep_crawling.scorers": {
                "KeywordRelevanceScorer": type("KeywordRelevanceScorer", (), {})
            },
            "crawl4ai.content_filter_strategy": {
                "BM25ContentFilter": type("BM25ContentFilter", (), {}),
                "PruningContentFilter": type("PruningContentFilter", (), {}),
            },
        }
    )

    crawling_pkg = _ensure_module("src.services.crawling")

    async def _noop_crawl_page(*_args: Any, **_kwargs: Any) -> Any:
        return {}

    _set_attributes(crawling_pkg, {"crawl_page": _noop_crawl_page})

    _register_crawl4ai_presets()
    _ensure_namespace_packages(("langchain_core", "langchain_community", "crawl4ai"))
    _register_contract_and_langgraph_stubs()


def _register_crawl4ai_presets() -> None:
    """Install Crawl4AI preset helpers used by the adapter tests."""

    presets_pkg = _ensure_module("src.services.crawling.c4a_presets")

    class _BrowserOptions:  # pragma: no cover - lightweight stub
        def __init__(
            self, browser_type: str = "chromium", headless: bool = True
        ) -> None:
            self.browser_type = browser_type
            self.headless = headless

    def _preset_browser_config(_: Any) -> Any:
        return types.SimpleNamespace()

    def _base_run_config(**kwargs: Any) -> Any:
        def clone(**clone_kwargs):
            return types.SimpleNamespace(**clone_kwargs)

        return types.SimpleNamespace(clone=clone, **kwargs)

    def _memory_dispatcher(**_kwargs: Any) -> Any:
        return object()

    _set_attributes(
        presets_pkg,
        {
            "BrowserOptions": _BrowserOptions,
            "preset_browser_config": _preset_browser_config,
            "base_run_config": _base_run_config,
            "memory_dispatcher": _memory_dispatcher,
        },
    )


def _register_contract_and_langgraph_stubs() -> None:
    """Install lightweight contract and LangGraph modules."""

    contracts_pkg = _ensure_module("contracts")
    retrieval_pkg = _ensure_module("contracts.retrieval")

    class _SearchRecord(BaseModel):  # pragma: no cover - minimal stub
        id: str = ""
        content: str = ""
        score: float = 0.0

        @classmethod
        def from_payload(cls, payload: Any) -> Self:
            if hasattr(payload, "model_dump"):
                data = payload.model_dump()
            elif isinstance(payload, Mapping):
                data = dict(payload)
            else:
                data = {}
            return cls(
                id=str(data.get("id", "")),
                content=str(data.get("content", "")),
                score=float(data.get("score", 0.0)),
            )

    _set_attributes(retrieval_pkg, {"SearchRecord": _SearchRecord})
    contracts_pkg.retrieval = retrieval_pkg  # type: ignore[attr-defined]

    flagembedding = _ensure_module("FlagEmbedding")
    _set_attributes(flagembedding, {})

    torch_module = _ensure_module("torch")
    if not hasattr(torch_module, "__path__"):
        torch_module.__path__ = []  # type: ignore[attr-defined]

    langgraph_pkg = _ensure_module("langgraph")
    graph_pkg = _ensure_module("langgraph.graph")
    _set_attributes(
        graph_pkg,
        {
            "StateGraph": type("StateGraph", (), {}),
            "END": object(),
        },
    )
    langgraph_pkg.graph = graph_pkg  # type: ignore[attr-defined]
    checkpoint_pkg = _ensure_module("langgraph.checkpoint")
    memory_pkg = _ensure_module("langgraph.checkpoint.memory")
    _set_attributes(memory_pkg, {"MemorySaver": type("MemorySaver", (), {})})
    checkpoint_pkg.memory = memory_pkg  # type: ignore[attr-defined]


def _register_miscellaneous_stubs() -> None:
    """Register additional third-party stubs referenced by the tests."""

    tiktoken = _ensure_module("tiktoken")
    _set_attributes(tiktoken, {})

    bs4_module = _ensure_module("bs4")
    _set_attributes(bs4_module, {"BeautifulSoup": type("BeautifulSoup", (), {})})


def register_rag_dependency_stubs() -> None:
    """Register all optional dependency stubs needed for the RAG tests."""

    _register_client_stubs()
    _register_langchain_core_stubs()
    _register_langchain_community_stubs()
    _register_crawl4ai_stubs()
    _register_miscellaneous_stubs()
