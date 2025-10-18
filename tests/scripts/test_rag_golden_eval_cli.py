"""Smoke tests for the RAG golden evaluation CLI script."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from scripts.eval.rag_golden_eval import (
    ExampleMetrics,
    ExampleResult,
    _build_arg_parser,
    _run,
)


pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnraisableExceptionWarning"
)


@pytest.fixture
def golden_dataset(tmp_path: Path) -> Path:
    """Write a minimal golden dataset for CLI smoke tests."""
    dataset = tmp_path / "golden.jsonl"
    row = {
        "query": "What is LangGraph?",
        "expected_answer": (
            "LangGraph is a state graph framework for LLM orchestration."
        ),
        "expected_contexts": [
            "LangGraph manages retrieval and generation via a declarative state graph."
        ],
        "references": ["https://example.com/langgraph"],
        "metadata": {"collection": "golden_eval"},
    }
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return dataset


class _StubOrchestrator:
    async def cleanup(self) -> None:
        return None


def _install_orchestrator_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_load_orchestrator() -> _StubOrchestrator:
        return _StubOrchestrator()

    async def _fake_evaluate_examples(
        examples,
        orchestrator,
        ragas_evaluator,
        *,  # keyword-only parameters to mirror real implementation
        limit,
        ragas_sample_cap=None,
        max_semantic_samples=None,
    ):
        _ = (
            orchestrator,
            ragas_evaluator,
            limit,
            ragas_sample_cap,
            max_semantic_samples,
        )
        metrics = ExampleMetrics(
            similarity=1.0,
            retrieval={
                "precision_at_k": 1.0,
                "recall_at_k": 1.0,
                "hit_rate": 1.0,
                "mrr": 1.0,
            },
            ragas={},
        )
        return [
            ExampleResult(
                example=examples[0],
                predicted_answer="Answer",
                answer_confidence=0.9,
                answer_sources=[{"doc_path": "docs/langgraph.md"}],
                processing_time_ms=10.0,
                metrics=metrics,
                records=[{"id": "doc-1"}],
            )
        ]

    monkeypatch.setattr(
        "scripts.eval.rag_golden_eval._load_orchestrator", _fake_load_orchestrator
    )
    monkeypatch.setattr(
        "scripts.eval.rag_golden_eval._evaluate_examples", _fake_evaluate_examples
    )

    @asynccontextmanager
    async def _fake_container_session(*_args, **_kwargs):
        yield

    monkeypatch.setattr(
        "scripts.eval.rag_golden_eval.container_session", _fake_container_session
    )


@pytest.mark.asyncio
async def test_cli_reports_success(
    tmp_path: Path, golden_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running the CLI without budgets should exit successfully and emit JSON."""
    _install_orchestrator_stub(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    output_path = tmp_path / "report.json"
    args = _build_arg_parser().parse_args(  # pylint: disable=protected-access
        [
            "--dataset",
            str(golden_dataset),
            "--output",
            str(output_path),
        ]
    )
    await _run(args)  # pylint: disable=protected-access

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["aggregates"]["examples"] == 1


@pytest.mark.asyncio
async def test_cli_budget_violation_raises(
    tmp_path: Path, golden_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Budgets below achievable values should cause a SystemExit(1)."""
    _install_orchestrator_stub(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "scripts.eval.rag_golden_eval._enforce_thresholds",
        lambda aggregates, thresholds: ["similarity"] if thresholds else [],
    )

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    budgets_path = config_dir / "eval_budgets.yml"
    budgets_path.write_text("similarity_avg: 0.99\n", encoding="utf-8")

    output_path = tmp_path / "report.json"
    args = _build_arg_parser().parse_args(  # pylint: disable=protected-access
        [
            "--dataset",
            str(golden_dataset),
            "--output",
            str(output_path),
        ]
    )

    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        await _run(args)  # pylint: disable=protected-access
    assert excinfo.value.code == 1
