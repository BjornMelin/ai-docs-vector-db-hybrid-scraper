"""Regression harness for the RAG pipeline against the golden dataset.

The harness executes the query processing pipeline with RAG enabled for each
entry in ``tests/data/rag/golden_set.jsonl`` and produces a lightweight
similarity score against the expected answer.  It is intentionally library
agnostic so the underlying implementation (LangChain/LangGraph) can change
without modifying this script.

Usage:
    uv run python scripts/eval/rag_golden_eval.py --dataset tests/data/rag/golden_set.jsonl

Notes:
    * The script requires a fully initialised QueryProcessingPipeline.  Provide
      the usual configuration (e.g., via environment variables) before running.
    * Scores are based on simple string similarity; replace ``_score_answer``
      with a richer metric (e.g., RAGAS, LangChain evaluators) once the
      LangChain migration lands.
"""  # noqa: E501

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.infrastructure.client_manager import ClientManager
from src.services.query_processing import QueryProcessingPipeline, SearchOrchestrator


@dataclass(slots=True)
class GoldenExample:
    """A single regression example."""

    query: str
    expected_answer: str
    references: list[str]
    metadata: dict[str, Any]


async def _load_pipeline() -> QueryProcessingPipeline:
    """Instantiate and initialize the shared query processing pipeline."""

    client_manager = ClientManager()
    vector_service = await client_manager.get_vector_store_service()
    orchestrator = SearchOrchestrator(vector_store_service=vector_service)
    pipeline = QueryProcessingPipeline(orchestrator)
    await pipeline.initialize()
    return pipeline


def _score_answer(predicted: str, expected: str) -> float:
    """Return a crude similarity score between predicted and expected text."""

    return SequenceMatcher(a=predicted.lower(), b=expected.lower()).ratio()


def _load_dataset(path: Path) -> list[GoldenExample]:
    """Read the golden dataset in JSON Lines format."""

    examples: list[GoldenExample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            examples.append(
                GoldenExample(
                    query=raw["query"],
                    expected_answer=raw["expected_answer"],
                    references=list(raw.get("references", [])),
                    metadata=dict(raw.get("metadata", {})),
                )
            )
    return examples


async def _run(dataset_path: Path) -> None:
    """Execute regression evaluation and print a short report."""

    examples = _load_dataset(dataset_path)
    if not examples:
        raise RuntimeError("Golden dataset is empty")

    pipeline = await _load_pipeline()

    scores: list[tuple[GoldenExample, float, str]] = []
    for example in examples:
        response = await pipeline.process(
            example.query,
            enable_rag=True,
            limit=5,
        )
        predicted = response.generated_answer or " ".join(
            record.content for record in response.records
        )
        score = _score_answer(predicted, example.expected_answer)
        scores.append((example, score, predicted))

    await pipeline.cleanup()

    print("=== RAG Golden Set Evaluation ===")
    for example, score, predicted in scores:
        print(f"Query: {example.query}")
        print(f"Expected: {example.expected_answer}")
        print(f"Predicted: {predicted}")
        print(f"Score: {score:.3f}")
        print(f"References: {example.references}")
        print("---")
    average = sum(score for _, score, _ in scores) / len(scores)
    print(f"Average similarity score: {average:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("tests/data/rag/golden_set.jsonl"),
        help="Path to the golden dataset (JSONL).",
    )
    args = parser.parse_args()

    asyncio.run(_run(args.dataset))


if __name__ == "__main__":
    main()
