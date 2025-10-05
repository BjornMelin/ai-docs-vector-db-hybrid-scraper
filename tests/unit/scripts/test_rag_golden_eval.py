"""Unit tests for the RAG golden evaluation harness."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.eval.rag_golden_eval import (
    EvaluationReport,
    ExampleMetrics,
    GoldenExample,
    RagasEvaluator,
    _aggregate_metrics,
    _evaluate_examples,
    _render_report,
)
from src.contracts.retrieval import SearchRecord
from src.services.query_processing.models import SearchResponse


class _StubOrchestrator:
    """Simple orchestrator stub returning canned responses."""

    def __init__(self, fixtures: dict[str, SearchResponse]) -> None:
        self._fixtures = fixtures

    async def search(
        self, request
    ) -> SearchResponse:  # pragma: no cover - small wrapper
        query = request.query
        try:
            return self._fixtures[query]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise AssertionError(f"Unexpected query: {query}") from exc

    async def cleanup(self) -> None:  # pragma: no cover - symmetrical no-op
        return None


class _DummyRagas(RagasEvaluator):
    """Ragas evaluator stub that returns deterministic values."""

    def __init__(self, score: float = 0.8) -> None:
        super().__init__(enabled=False)
        self._score = score

    def evaluate(
        self,
        example: GoldenExample,
        predicted_answer: str,
        contexts: Any,
    ) -> dict[str, float]:  # type: ignore[override]
        return {"faithfulness": self._score}


@pytest.mark.asyncio
async def test_evaluate_examples_and_aggregation() -> None:
    """The evaluator computes deterministic, reproducible metrics."""

    examples = [
        GoldenExample(
            query="How do I rotate API keys?",
            expected_answer=(
                "Use the security portal to generate a new key, "
                "update service configs, then revoke the old key."
            ),
            expected_contexts=[
                (
                    "Rotate keys by issuing a replacement in the security portal, "
                    "updating all services with the new credential, and removing "
                    "the old key immediately after verifying connectivity."
                )
            ],
            references=["docs/security/api-keys.md"],
            metadata={"collection": "docs"},
        ),
        GoldenExample(
            query="What is the default grouping field?",
            expected_answer=(
                "Results are grouped by the `doc_id` payload unless overridden "
                "per request."
            ),
            expected_contexts=[
                (
                    "Grouping uses the `doc_id` field by default so deduped "
                    "passages from the same document collapse into a single "
                    "group unless the caller supplies a different payload key."
                )
            ],
            references=["docs/query_processing_response_contract.md"],
            metadata={"collection": "docs"},
        ),
    ]

    fixtures: dict[str, SearchResponse] = {
        examples[0].query: SearchResponse(
            records=[
                SearchRecord(
                    id="doc-1",
                    content=(
                        "Rotate keys via the security portal then revoke the old "
                        "credentials"
                    ),
                    score=0.99,
                    metadata={"doc_path": "docs/security/api-keys.md"},
                )
            ],
            total_results=1,
            query=examples[0].query,
            processing_time_ms=42.0,
            generated_answer=(
                "Generate a replacement key in the security portal, roll it out, "
                "then delete the old one."
            ),
            answer_confidence=0.85,
            answer_sources=[{"doc_path": "docs/security/api-keys.md"}],
        ),
        examples[1].query: SearchResponse(
            records=[
                SearchRecord(
                    id="doc-2",
                    content=(
                        "By default results are grouped on the doc_id payload field"
                    ),
                    score=0.92,
                    metadata={"doc_path": "docs/query_processing_response_contract.md"},
                )
            ],
            total_results=1,
            query=examples[1].query,
            processing_time_ms=55.0,
            generated_answer=(
                "The orchestrator groups on the doc_id payload unless callers "
                "supply a different key."
            ),
            answer_confidence=0.9,
            answer_sources=[{"doc_path": "docs/query_processing_response_contract.md"}],
        ),
    }

    orchestrator = _StubOrchestrator(fixtures)
    ragas_evaluator = _DummyRagas(score=0.75)

    results = await _evaluate_examples(
        examples,
        orchestrator,
        ragas_evaluator,
        limit=5,
    )

    assert len(results) == 2
    for result in results:
        assert isinstance(result.metrics, ExampleMetrics)
        assert result.metrics.similarity > 0.45
        assert result.metrics.retrieval["precision_at_k"] == pytest.approx(1.0)
        assert result.metrics.retrieval["recall_at_k"] == pytest.approx(1.0)
        assert result.metrics.retrieval["hit_rate"] == 1.0
        assert result.metrics.retrieval["mrr"] == pytest.approx(1.0)
        assert result.metrics.ragas["faithfulness"] == pytest.approx(0.75)

    aggregates = _aggregate_metrics(results)
    assert aggregates["examples"] == 2
    assert aggregates["similarity_avg"] > 0.5
    assert aggregates["processing_time_ms_avg"] == pytest.approx(48.5)
    assert aggregates["retrieval_avg"]["precision_at_k"] == pytest.approx(1.0)
    assert aggregates["ragas_avg"]["faithfulness"] == pytest.approx(0.75)

    report = _render_report(
        EvaluationReport(results=results, aggregates=aggregates, telemetry={})
    )
    assert len(report["results"]) == 2
    assert report["aggregates"]["examples"] == 2


@pytest.mark.asyncio
async def test_evaluate_examples_handles_missing_records() -> None:
    """Empty retrieval results produce zeroed metrics without crashing."""

    example = GoldenExample(
        query="Missing context",
        expected_answer="No answer",
        expected_contexts=[],
        references=["docs/missing.md"],
        metadata={},
    )

    fixtures = {
        example.query: SearchResponse(
            records=[],
            total_results=0,
            query=example.query,
            processing_time_ms=5.0,
            generated_answer="",
            answer_confidence=None,
            answer_sources=[],
        )
    }

    orchestrator = _StubOrchestrator(fixtures)
    ragas_evaluator = _DummyRagas(score=0.5)

    results = await _evaluate_examples(
        [example],
        orchestrator,
        ragas_evaluator,
        limit=3,
    )

    assert results[0].metrics.retrieval["precision_at_k"] == 0.0
    assert results[0].metrics.retrieval["recall_at_k"] == 0.0
    assert results[0].metrics.retrieval["hit_rate"] == 0.0
    assert results[0].metrics.retrieval["mrr"] == 0.0
