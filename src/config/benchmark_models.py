"""Pydantic models for benchmark configuration files.

This module defines the schema for custom-benchmarks.json and related
benchmark configuration files used in the embedding system.
"""

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .models import ModelBenchmark
from .models import SmartSelectionConfig


class EmbeddingBenchmarkSet(BaseModel):
    """Complete benchmark configuration for embedding models.

    This model represents the structure of custom-benchmarks.json file,
    containing smart selection parameters and model benchmarks.
    """

    smart_selection: SmartSelectionConfig = Field(
        description="Configuration for smart model selection algorithms"
    )
    model_benchmarks: dict[str, ModelBenchmark] = Field(
        description="Performance benchmarks for embedding models, keyed by model name"
    )

    model_config = ConfigDict(extra="forbid")


class BenchmarkConfiguration(BaseModel):
    """Root configuration for benchmark files.

    This is the top-level model for custom-benchmarks.json structure,
    with nested embedding configuration.
    """

    embedding: EmbeddingBenchmarkSet = Field(
        description="Embedding benchmark configuration"
    )

    model_config = ConfigDict(extra="forbid")
