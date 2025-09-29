"""Tests for the deterministic processing helpers used by embedding pipelines."""

from __future__ import annotations

import asyncio

import pytest

from src.services.processing.algorithms import OptimizedTextAnalyzer
from src.services.processing.batch_optimizer import BatchConfig, BatchProcessor


@pytest.mark.anyio
async def test_batch_processor_flushes_on_batch_size() -> None:
    """Batches should be processed immediately once the size threshold is met."""
    processed: list[list[str]] = []

    def _process(items: list[str]) -> list[str]:
        processed.append(list(items))
        return [item.upper() for item in items]

    processor = BatchProcessor(
        _process,
        BatchConfig(
            max_batch_size=2, min_batch_size=1, max_wait_time=1.0, adaptive_sizing=False
        ),
    )

    results = await asyncio.gather(
        processor.process_item("dense"),
        processor.process_item("hybrid"),
    )

    assert results == ["DENSE", "HYBRID"]
    assert processed == [["dense", "hybrid"]]
    assert processor.batch_performance_history
    last_batch_size, _ = processor.batch_performance_history[-1]
    assert last_batch_size == 2


@pytest.mark.anyio
async def test_batch_processor_propagates_exceptions() -> None:
    """Errors from the processing function should bubble to awaiting callers."""

    def _failing(_items: list[str]) -> list[str]:
        msg = "processing failed"
        raise RuntimeError(msg)

    processor = BatchProcessor(_failing, BatchConfig(max_batch_size=1))

    with pytest.raises(RuntimeError):
        await processor.process_item("payload")


def test_text_analyzer_reports_keyword_density() -> None:
    """Optimized analyzer should extract keywords from hybrid-search copy."""
    analyzer = OptimizedTextAnalyzer()
    text = (
        "Vector search powers agentic workflows. "
        "Vector search blends dense and sparse signals."
    )

    result = analyzer.analyze_text_optimized(text)
    assert result.word_count > 0
    assert result.sentence_count == 2
    assert result.keyword_density["vector"] > 0
    assert 0.0 <= result.complexity_score <= 1.0
    assert result.processing_time_ms >= 0


def test_text_analyzer_handles_empty_input() -> None:
    """Empty input should return zeroed metrics without raising."""
    analyzer = OptimizedTextAnalyzer()
    result = analyzer.analyze_text_optimized("")
    assert result.word_count == 0
    assert not result.keyword_density
    assert result.processing_time_ms == 0.0
