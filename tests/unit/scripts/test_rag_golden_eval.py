"""Unit tests for the RAG golden evaluation harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.eval.dataset_validator import DatasetValidationError, validate_dataset
from scripts.eval.rag_golden_eval import (
    EvaluationReport,
    ExampleMetrics,
    GoldenExample,
    RagasEvaluator,
    _aggregate_metrics,
    _enforce_thresholds,
    _evaluate_examples,
    _load_cost_controls,
    _load_dataset,
    _load_thresholds,
    _render_report,
)
from src.contracts.retrieval import SearchRecord, SearchResponse


class _StubOrchestrator:
    """Simple orchestrator stub returning canned responses."""

    def __init__(self, fixtures: dict[str, SearchResponse]) -> None:
        """Initialize the stub orchestrator with predefined fixtures."""
        self._fixtures = fixtures

    async def search(self, request) -> SearchResponse:  # pragma: no cover - wrapper
        """Return a canned search response based on the query."""
        query = request.query
        try:
            return self._fixtures[query]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise AssertionError(f"Unexpected query: {query}") from exc

    async def cleanup(self) -> None:  # pragma: no cover - symmetrical no-op
        """Perform any necessary cleanup after search operations."""
        return


class _DummyRagas(RagasEvaluator):
    """Ragas evaluator stub that returns deterministic values."""

    def __init__(self, score: float = 0.8) -> None:
        """Initialize the dummy evaluator with a fixed score."""
        # pylint: disable=super-init-not-called
        self._score = score

    def evaluate(  # type: ignore[override]
        self,
        example: GoldenExample,
        predicted_answer: str,
        contexts: Any,
    ) -> dict[str, float]:
        """Return a fixed faithfulness score."""
        return {"faithfulness": self._score}


@pytest.mark.asyncio
async def test_evaluate_examples_and_aggregation() -> None:
    """The evaluator computes deterministic, reproducible metrics."""
    examples = [
        GoldenExample(
            query="How should operators rotate API keys?",
            expected_answer=(
                "Generate a replacement key in the security portal, roll it "
                "out to dependent services, validate traffic, and immediately "
                "revoke the old key."
            ),
            expected_contexts=[
                (
                    "To rotate API keys, issue a new credential in the security "
                    "portal, update every dependent service, verify connectivity, "
                    "and revoke the retired key immediately after validation."
                )
            ],
            references=["docs/security/security-essentials.md#api-key-rotation"],
            metadata={"collection": "golden_eval"},
        ),
        GoldenExample(
            query="What payload field is used to group search results by default?",
            expected_answer=(
                "Results are grouped by the doc_id payload unless the request "
                "overrides group_by."
            ),
            expected_contexts=[
                (
                    "Server-side grouping collapses results that share the same "
                    "doc_id payload value unless the caller provides a custom "
                    "group_by key."
                )
            ],
            references=["docs/query/evaluation-reference.md#grouping-defaults"],
            metadata={"collection": "golden_eval"},
        ),
    ]

    fixtures: dict[str, SearchResponse] = {
        examples[0].query: SearchResponse(
            records=[
                SearchRecord(
                    id="doc-1",
                    content=(
                        "To rotate API keys, issue a new credential in the security "
                        "portal, update every dependent service, verify connectivity, "
                        "and revoke the retired key immediately after validation."
                    ),
                    score=0.99,
                    metadata={
                        "doc_path": (
                            "docs/security/security-essentials.md#api-key-rotation"
                        )
                    },
                )
            ],
            total_results=1,
            query=examples[0].query,
            processing_time_ms=42.0,
            generated_answer=(
                "Generate a replacement key in the security portal, roll it out to "
                "dependent services, validate traffic, and immediately revoke the "
                "old key."
            ),
            answer_confidence=0.85,
            answer_sources=[
                {"doc_path": ("docs/security/security-essentials.md#api-key-rotation")}
            ],
        ),
        examples[1].query: SearchResponse(
            records=[
                SearchRecord(
                    id="doc-2",
                    content=(
                        "Server-side grouping collapses results that share the same "
                        "doc_id payload value unless the caller provides a custom "
                        "group_by key."
                    ),
                    score=0.92,
                    metadata={
                        "doc_path": (
                            "docs/query/evaluation-reference.md#grouping-defaults"
                        )
                    },
                )
            ],
            total_results=1,
            query=examples[1].query,
            processing_time_ms=55.0,
            generated_answer=(
                "Results are grouped by the doc_id payload unless the request "
                "overrides group_by."
            ),
            answer_confidence=0.9,
            answer_sources=[
                {"doc_path": "docs/query/evaluation-reference.md#grouping-defaults"}
            ],
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
        assert result.metrics.similarity > 0.9
        assert result.metrics.retrieval["precision_at_k"] == pytest.approx(1.0)
        assert result.metrics.retrieval["recall_at_k"] == pytest.approx(1.0)
        assert result.metrics.retrieval["hit_rate"] == 1.0
        assert result.metrics.retrieval["mrr"] == pytest.approx(1.0)
        assert result.metrics.ragas["faithfulness"] == pytest.approx(0.75)

    aggregates = _aggregate_metrics(results)
    assert aggregates["examples"] == 2
    assert aggregates["similarity_avg"] > 0.9
    assert aggregates["processing_time_ms_avg"] == pytest.approx(48.5)
    assert aggregates["retrieval_avg"]["precision_at_k"] == pytest.approx(1.0)
    assert aggregates["ragas_avg"]["faithfulness"] == pytest.approx(0.75)

    report = _render_report(EvaluationReport(results=results, aggregates=aggregates))
    assert len(report["results"]) == 2
    assert report["aggregates"]["examples"] == 2


@pytest.mark.asyncio
async def test_evaluate_examples_handles_missing_records() -> None:
    """Empty retrieval results produce zeroed metrics without crashing."""
    example = GoldenExample(
        query="Missing context",
        expected_answer="No answer",
        expected_contexts=[],
        references=["docs/testing/missing.md"],
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


def test_load_dataset_raises_on_invalid_json(tmp_path: Path) -> None:
    """Malformed JSON should raise a descriptive error."""
    dataset_path = tmp_path / "broken.jsonl"
    dataset_path.write_text("{invalid json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _load_dataset(dataset_path)


def test_dataset_validator_detects_missing_metadata(tmp_path: Path) -> None:
    """Dataset validator should fail when required metadata keys are absent."""
    dataset_path = tmp_path / "dataset.jsonl"
    record = {
        "query": "test",
        "expected_answer": "answer",
        "expected_contexts": [],
        "references": [],
        "metadata": {},
    }
    dataset_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(DatasetValidationError):
        validate_dataset(dataset_path)


def test_enforce_thresholds_detects_budget_failures() -> None:
    """Threshold enforcement should identify metrics that exceed budgets."""
    aggregates = {
        "similarity_avg": 0.72,
        "processing_time_ms_avg": 1800.0,
        "retrieval_avg": {"precision_at_k": 0.75, "recall_at_k": 0.82},
    }
    thresholds = {
        "similarity_avg": 0.75,
        "precision_at_k": 0.8,
        "recall_at_k": 0.8,
        "max_latency_ms": 1500,
    }

    failures = _enforce_thresholds(aggregates, thresholds)
    assert "similarity_avg" in failures[0]
    assert any("precision_at_k" in failure for failure in failures)
    assert any("processing_time_ms_avg" in failure for failure in failures)


def test_load_cost_controls_handles_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing config files should yield None and empty config."""
    monkeypatch.chdir(tmp_path)
    max_samples, config = _load_cost_controls()
    assert max_samples is None
    assert config == {}


def test_load_thresholds_filters_numeric_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Threshold loader should ignore non-numeric entries and coerce floats."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    budgets_path = config_dir / "eval_budgets.yml"
    budgets_path.write_text(
        """similarity_avg: 0.8
precision_at_k: 1
recall_at_k: not-a-number
max_latency_ms: 1500.5
""",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    thresholds = _load_thresholds()
    assert thresholds == {
        "similarity_avg": 0.8,
        "precision_at_k": 1.0,
        "max_latency_ms": 1500.5,
    }


def test_enforce_thresholds_returns_empty_list_when_no_thresholds() -> None:
    """No configured thresholds should produce no failures."""
    assert not _enforce_thresholds({"similarity_avg": 1.0}, {})


def test_load_cost_controls_coerces_integer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cost control loader should parse integer max samples."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    costs_path = config_dir / "eval_costs.yml"
    costs_path.write_text("semantic:\n  max_samples: 12\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    max_samples, config = _load_cost_controls()
    assert max_samples == 12
    assert config["semantic"]["max_samples"] == 12
