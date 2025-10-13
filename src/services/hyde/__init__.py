"""HyDE (Hypothetical Document Embeddings) implementation."""

from .config import HyDEConfig
from .engine import HyDEQueryEngine
from .generator import HypotheticalDocumentGenerator


__all__ = [
    "HyDEConfig",
    "HyDEQueryEngine",
    "HypotheticalDocumentGenerator",
]
