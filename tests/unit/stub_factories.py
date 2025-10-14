"""Utilities for registering dependency stubs used in unit tests."""

from __future__ import annotations

import sys
import types
from collections.abc import Iterable
from typing import Any, cast


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


def _register_client_stubs() -> None:
    """Register stubs for client SDKs leveraged by the RAG pipeline."""
    firecrawl = _ensure_module("firecrawl")
    _set_attributes(
        firecrawl,
        {"AsyncFirecrawlApp": type("AsyncFirecrawlApp", (), {})},
    )

    openai = _ensure_module("openai")
    _set_attributes(openai, {"AsyncOpenAI": type("AsyncOpenAI", (), {})})

    langchain_openai = _ensure_module("langchain_openai")
    _set_attributes(langchain_openai, {"ChatOpenAI": type("ChatOpenAI", (), {})})

    qdrant = _ensure_module("langchain_qdrant")
    _set_attributes(
        qdrant,
        {
            "QdrantVectorStore": type("QdrantVectorStore", (), {}),
            "RetrievalMode": type("RetrievalMode", (), {}),
            "FastEmbedSparse": type("FastEmbedSparse", (), {}),
        },
    )


def _register_langchain_core_stubs() -> None:
    """Register core LangChain modules required for RAG tests."""
    langchain = cast(Any, _ensure_module("langchain"))
    retrievers = cast(Any, _ensure_module("langchain.retrievers"))
    document_compressors = cast(
        Any, _ensure_module("langchain.retrievers.document_compressors")
    )
    _set_attributes(
        document_compressors,
        {
            "DocumentCompressorPipeline": type("DocumentCompressorPipeline", (), {}),
            "EmbeddingsFilter": type("EmbeddingsFilter", (), {}),
        },
    )
    retrievers.document_compressors = document_compressors
    langchain.retrievers = retrievers

    callbacks_manager = cast(Any, _ensure_module("langchain_core.callbacks.manager"))
    _set_attributes(
        callbacks_manager,
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

    messages = cast(Any, _ensure_module("langchain_core.messages"))
    _set_attributes(
        messages,
        {
            "AIMessage": type("AIMessage", (), {}),
            "HumanMessage": type("HumanMessage", (), {}),
        },
    )

    tools = cast(Any, _ensure_module("langchain_core.tools"))
    if not hasattr(tools, "__path__"):
        tools.__path__ = []  # type: ignore[attr-defined]
    _set_attributes(
        tools,
        {
            "BaseTool": type("BaseTool", (), {}),
            "InjectedToolArg": type("InjectedToolArg", (), {}),
            "StructuredTool": type("StructuredTool", (), {}),
            "Tool": type("Tool", (), {}),
            "ToolException": type("ToolException", (), {}),
        },
    )

    tools_base = cast(Any, _ensure_module("langchain_core.tools.base"))
    _set_attributes(
        tools_base,
        {
            "BaseTool": type("BaseTool", (), {}),
            "get_all_basemodel_annotations": lambda *args, **kwargs: {},
        },
    )

    prompts = cast(Any, _ensure_module("langchain_core.prompts"))
    _set_attributes(prompts, {"ChatPromptTemplate": _ChatPromptTemplate})

    documents = cast(Any, _ensure_module("langchain_core.documents"))
    _set_attributes(documents, {"Document": type("Document", (), {})})

    documents_base = cast(Any, _ensure_module("langchain_core.documents.base"))
    _set_attributes(documents_base, {"Blob": type("Blob", (), {})})

    retrievers_core = cast(Any, _ensure_module("langchain_core.retrievers"))
    _set_attributes(retrievers_core, {"BaseRetriever": type("BaseRetriever", (), {})})

    text_splitters = cast(Any, _ensure_module("langchain_text_splitters"))
    _set_attributes(
        text_splitters,
        {
            "RecursiveCharacterTextSplitter": type(
                "RecursiveCharacterTextSplitter", (), {}
            ),
        },
    )

    if not hasattr(langchain, "__path__"):
        langchain.__path__ = []  # type: ignore[attr-defined]


def _register_langchain_community_stubs() -> None:
    """Register LangChain community modules used by the RAG pipeline."""
    community = cast(Any, _ensure_module("langchain_community"))
    community_embeddings = cast(Any, _ensure_module("langchain_community.embeddings"))
    fastembed = cast(Any, _ensure_module("langchain_community.embeddings.fastembed"))
    _set_attributes(
        fastembed,
        {"FastEmbedEmbeddings": type("FastEmbedEmbeddings", (), {})},
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
    crawl4ai = _ensure_module("crawl4ai")
    _set_attributes(
        crawl4ai,
        {
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
    )

    crawl4ai_models = _ensure_module("crawl4ai.models")
    _set_attributes(crawl4ai_models, {"CrawlResult": type("CrawlResult", (), {})})

    async_dispatcher = _ensure_module("crawl4ai.async_dispatcher")
    _set_attributes(
        async_dispatcher, {"SemaphoreDispatcher": type("SemaphoreDispatcher", (), {})}
    )

    crawl4ai_deep = _ensure_module("crawl4ai.deep_crawling")
    _set_attributes(
        crawl4ai_deep,
        {
            "BestFirstCrawlingStrategy": type("BestFirstCrawlingStrategy", (), {}),
            "BFSDeepCrawlStrategy": type("BFSDeepCrawlStrategy", (), {}),
        },
    )

    filters_module = _ensure_module("crawl4ai.deep_crawling.filters")
    _set_attributes(
        filters_module,
        {
            "ContentRelevanceFilter": type("ContentRelevanceFilter", (), {}),
            "ContentTypeFilter": type("ContentTypeFilter", (), {}),
            "DomainFilter": type("DomainFilter", (), {}),
            "FilterChain": type("FilterChain", (), {}),
            "SEOFilter": type("SEOFilter", (), {}),
            "URLFilter": type("URLFilter", (), {}),
            "URLPatternFilter": type("URLPatternFilter", (), {}),
        },
    )

    scorers_module = _ensure_module("crawl4ai.deep_crawling.scorers")
    _set_attributes(
        scorers_module,
        {"KeywordRelevanceScorer": type("KeywordRelevanceScorer", (), {})},
    )

    content_filter = _ensure_module("crawl4ai.content_filter_strategy")
    _set_attributes(
        content_filter,
        {
            "BM25ContentFilter": type("BM25ContentFilter", (), {}),
            "PruningContentFilter": type("PruningContentFilter", (), {}),
        },
    )

    for namespace in ("langchain_core", "langchain_community", "crawl4ai"):
        package = _ensure_module(namespace)
        if not hasattr(package, "__path__"):
            package.__path__ = []  # type: ignore[attr-defined]


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
