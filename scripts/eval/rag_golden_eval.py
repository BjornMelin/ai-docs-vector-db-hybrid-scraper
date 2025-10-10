"""Evaluate the RAG pipeline against the golden dataset with structured metrics.

This script executes the LangGraph-backed search orchestrator for every
entry in the golden dataset and captures three layers of evaluation:

* Deterministic baselines (string similarity + retrieval precision/recall)
* Optional Ragas metrics (faithfulness, relevance, context recall)

Usage:
    uv run python scripts/eval/rag_golden_eval.py \
        --dataset tests/data/rag/golden_set.jsonl \
        --output artifacts/rag_golden_report.json

Enable semantic metrics by providing an API-enabled LLM/embedding via
environment variables and passing ``--enable-ragas``.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

import yaml

from scripts.eval.dataset_validator import DatasetValidationError, load_dataset_records
from src.config import get_settings
from src.infrastructure.container import (
    get_container,
    initialize_container,
    shutdown_container,
)
from src.services.query_processing.models import SearchRequest, SearchResponse
from src.services.query_processing.orchestrator import SearchOrchestrator


LOGGER = logging.getLogger(__name__)

# Ragas imports are optional to keep the harness runnable in offline mode.
# They are resolved lazily inside ``RagasEvaluator``.


@dataclass(slots=True)
class GoldenExample:
    """Input row for the regression evaluation harness."""

    query: str
    expected_answer: str
    expected_contexts: list[str]
    references: list[str]
    metadata: dict[str, Any]


@dataclass(slots=True)
class ExampleMetrics:
    """Metrics captured for an individual evaluation example."""

    similarity: float
    retrieval: dict[str, float]
    ragas: dict[str, float]


@dataclass(slots=True)
class ExampleResult:
    """Outcome of evaluating a single golden dataset example."""

    example: GoldenExample
    predicted_answer: str
    answer_confidence: float | None
    answer_sources: list[dict[str, Any]]
    processing_time_ms: float
    metrics: ExampleMetrics
    records: list[dict[str, Any]]


@dataclass(slots=True)
class EvaluationReport:
    """Complete evaluation artefact emitted by the harness."""

    results: list[ExampleResult]
    aggregates: dict[str, Any]


class SupportsSearchOrchestrator(Protocol):
    """Protocol describing the orchestrator surface required by the harness."""

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute a search request."""
        raise NotImplementedError

    async def cleanup(self) -> None:
        """Release any orchestrator resources."""
        raise NotImplementedError


class RagasEvaluator:
    """Wrapper around Ragas metrics with graceful degradation when disabled."""

    def __init__(
        self,
        *,
        enabled: bool,
        llm_model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self._enabled = False
        self._evaluate = None
        self._metrics_suite: list[Any] = []
        self._llm = None
        self._embeddings = None
        self._dataset_cls: Any | None = None
        if not enabled:
            LOGGER.info("Ragas evaluation disabled; skipping semantic metrics.")
            return

        self._enabled = self._setup_backend(llm_model, embedding_model)

    def _setup_backend(
        self, llm_model: str | None, embedding_model: str | None
    ) -> bool:
        """Initialise optional Ragas dependencies."""

        # pylint: disable=import-outside-toplevel,too-many-locals
        try:  # pragma: no cover - optional dependency import path
            langchain_openai = importlib.import_module("langchain_openai")
            ragas_module = importlib.import_module("ragas")
            dataset_schema = importlib.import_module("ragas.dataset_schema")
            embeddings_module = importlib.import_module("ragas.embeddings")
            llms_module = importlib.import_module("ragas.llms")
            metrics_module = importlib.import_module("ragas.metrics")

            chat_open_ai = langchain_openai.ChatOpenAI
            openai_embeddings = langchain_openai.OpenAIEmbeddings
            self._dataset_cls = dataset_schema.EvaluationDataset
            self._evaluate = ragas_module.evaluate
            self._metrics_suite = [
                metrics_module.context_precision,
                metrics_module.context_recall,
                metrics_module.faithfulness,
                metrics_module.answer_relevancy,
            ]
            llm_instance = chat_open_ai(model=llm_model or "gpt-4o-mini")
            embedding_instance = openai_embeddings(
                model=embedding_model or "text-embedding-3-small"
            )
            embeddings_wrapper = embeddings_module.LangchainEmbeddingsWrapper
            llm_wrapper = llms_module.LangchainLLMWrapper
            self._llm = llm_wrapper(llm_instance)
            self._embeddings = embeddings_wrapper(embedding_instance)
            LOGGER.info(
                "Ragas evaluator initialised with model=%s embedding=%s",
                llm_model or "gpt-4o-mini",
                embedding_model or "text-embedding-3-small",
            )
            return True
        except ImportError as exc:  # pragma: no cover - optional dependency path
            LOGGER.warning(
                "ragas/langchain_openai missing; semantic metrics disabled (%s)",
                exc,
            )
            return False
        except Exception as exc:  # pragma: no cover - environment specific
            LOGGER.warning(
                "Failed to initialise Ragas evaluator, metrics disabled: %s",
                exc,
                exc_info=True,
            )
            return False

    def evaluate(
        self,
        example: GoldenExample,
        predicted_answer: str,
        contexts: Sequence[str],
    ) -> dict[str, float]:
        """Return semantic metrics using Ragas when enabled."""

        if not self._enabled or self._evaluate is None or self._dataset_cls is None:
            return {}

        try:  # pragma: no cover - networked path
            dataset = self._dataset_cls.from_list(
                [
                    {
                        "question": example.query,
                        "answer": predicted_answer,
                        "contexts": list(contexts),
                        "ground_truth": [example.expected_answer],
                        "reference_contexts": list(example.expected_contexts or []),
                    }
                ]
            )
            result = self._evaluate(
                dataset=dataset,
                metrics=self._metrics_suite,
                llm=self._llm,
                embeddings=self._embeddings,
            )

            raw_scores: dict[str, float] = {}
            for key, value in self._collect_metric_samples(result):
                if value is None:
                    continue
                try:
                    raw_scores[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

            clean_scores: dict[str, float] = {}
            for metric in self._metrics_suite:
                metric_name = getattr(metric, "name", None)
                if not metric_name:
                    continue
                for candidate_key in (
                    metric_name,
                    f"{metric_name}_score",
                    metric_name.replace("-", "_"),
                ):
                    if candidate_key in raw_scores:
                        clean_scores[metric_name] = raw_scores[candidate_key]
                        break
            return clean_scores
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.warning(
                "Ragas evaluation failed for query '%s': %s",
                example.query,
                exc,
                exc_info=True,
            )
            return {}

    @staticmethod
    def _collect_metric_samples(raw_result: Any) -> Iterable[tuple[str, Any]]:
        """Normalise the various result payloads emitted by Ragas."""

        if hasattr(raw_result, "metrics") and isinstance(raw_result.metrics, dict):
            return raw_result.metrics.items()
        if hasattr(raw_result, "scores") and isinstance(raw_result.scores, dict):
            return raw_result.scores.items()
        if hasattr(raw_result, "to_dict"):
            payload = raw_result.to_dict()
            if isinstance(payload, dict):
                return payload.items()
        if isinstance(raw_result, dict):
            return raw_result.items()
        return []


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file returning an empty dict when missing or invalid."""

    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to load YAML config at %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        LOGGER.warning("YAML config at %s is not a mapping; ignoring", path)
        return {}
    return data


def _load_cost_controls() -> tuple[int | None, dict[str, Any]]:
    """Return semantic evaluation caps and raw config."""

    config = _load_yaml(Path("config/eval_costs.yml"))
    raw_semantic = config.get("semantic")
    semantic = raw_semantic if isinstance(raw_semantic, dict) else {}
    max_samples_value = semantic.get("max_samples")
    max_samples = max_samples_value if isinstance(max_samples_value, int) else None
    return max_samples, config


def _load_thresholds() -> dict[str, float]:
    """Load evaluation threshold budget values."""

    raw = _load_yaml(Path("config/eval_budgets.yml"))
    thresholds: dict[str, float] = {}
    for key in ("similarity_avg", "precision_at_k", "recall_at_k", "max_latency_ms"):
        value = raw.get(key)
        if isinstance(value, int | float):
            thresholds[key] = float(value)
    return thresholds


def _format_threshold_failure(metric: str, observed: float, required: float) -> str:
    """Return a concise message describing a threshold violation."""

    return f"{metric} {observed:.3f} outside budget {required:.3f}"


def _enforce_thresholds(
    aggregates: dict[str, Any], thresholds: dict[str, float]
) -> list[str]:
    """Return failures when aggregates fall outside the configured budgets."""

    failures: list[str] = []
    if not thresholds:
        return failures

    if "similarity_avg" in thresholds:
        similarity = float(aggregates.get("similarity_avg", 0.0) or 0.0)
        required = thresholds["similarity_avg"]
        if similarity < required:
            failures.append(
                _format_threshold_failure("similarity_avg", similarity, required)
            )

    retrieval_avg = aggregates.get("retrieval_avg", {})
    if not isinstance(retrieval_avg, dict):
        retrieval_avg = {}

    for key in ("precision_at_k", "recall_at_k"):
        if key in thresholds:
            observed = float(retrieval_avg.get(key, 0.0) or 0.0)
            required = thresholds[key]
            if observed < required:
                failures.append(_format_threshold_failure(key, observed, required))

    if "max_latency_ms" in thresholds:
        latency = float(aggregates.get("processing_time_ms_avg", 0.0) or 0.0)
        required_latency = thresholds["max_latency_ms"]
        if latency > required_latency:
            failures.append(
                _format_threshold_failure(
                    "processing_time_ms_avg",
                    latency,
                    required_latency,
                )
            )

    return failures


def _load_dataset(path: Path) -> list[GoldenExample]:
    """Parse the golden dataset in JSON Lines format."""

    if not path.exists():
        msg = f"Dataset path does not exist: {path}"
        raise FileNotFoundError(msg)

    try:
        rows = load_dataset_records(path)
    except DatasetValidationError as exc:
        raise ValueError(str(exc)) from exc
    if not rows:
        raise RuntimeError("Golden dataset is empty")

    return [
        GoldenExample(
            query=str(payload["query"]),
            expected_answer=str(payload["expected_answer"]),
            expected_contexts=list(payload.get("expected_contexts", [])),
            references=list(payload.get("references", [])),
            metadata=dict(payload.get("metadata", {})),
        )
        for payload in rows
    ]


async def _load_orchestrator() -> SearchOrchestrator:
    """Instantiate and initialise the shared search orchestrator."""

    container = get_container()
    if container is None:
        container = await initialize_container(get_settings())
    vector_service = container.vector_store_service()
    if vector_service is None:
        raise RuntimeError("Vector store service unavailable")
    orchestrator = SearchOrchestrator(vector_store_service=vector_service)
    await orchestrator.initialize()
    return orchestrator


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(candidate: str) -> str:
    """Return a lowercase, normalised representation for similarity scoring."""

    collapsed = _WHITESPACE_RE.sub(" ", candidate.lower()).strip()
    return collapsed


def _compute_similarity(predicted: str, expected: str) -> float:
    """Deterministic string similarity baseline."""

    matcher = SequenceMatcher(
        a=_normalize_text(predicted),
        b=_normalize_text(expected),
    )
    return matcher.ratio()


def _extract_reference(record: dict[str, Any]) -> str | None:
    """Best-effort extraction of a reference identifier from a search record."""

    metadata = record.get("metadata") or {}
    for key in ("doc_path", "source_path", "reference", "uri"):
        candidate = metadata.get(key)
        if candidate:
            return str(candidate)
    if record.get("group_id"):
        return str(record["group_id"])
    return record.get("id")


def _compute_retrieval_metrics(
    example: GoldenExample,
    records: Sequence[dict[str, Any]],
    *,
    k: int,
) -> dict[str, float]:
    """Precision/recall style metrics derived from retrieved records."""

    if not records:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "hit_rate": 0.0, "mrr": 0.0}

    expected = {ref.lower() for ref in example.references}
    ranked_refs = [_extract_reference(record) for record in records]
    ranked_refs = [ref.lower() for ref in ranked_refs if ref]

    top_k = ranked_refs[: max(1, k)]
    hits = [ref for ref in top_k if ref in expected]

    precision = len(hits) / len(top_k) if top_k else 0.0
    recall = len(hits) / len(expected) if expected else 0.0
    hit_rate = 1.0 if hits else 0.0

    reciprocal_rank = 0.0
    for index, ref in enumerate(ranked_refs, start=1):
        if ref in expected:
            reciprocal_rank = 1.0 / index
            break

    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "hit_rate": hit_rate,
        "mrr": reciprocal_rank,
    }


def _aggregate_metrics(results: Sequence[ExampleResult]) -> dict[str, Any]:
    """Aggregate numeric metrics across all examples."""

    if not results:
        return {}

    similarity_scores = [item.metrics.similarity for item in results]

    retrieval_bucket: dict[str, list[float]] = defaultdict(list)
    ragas_bucket: dict[str, list[float]] = defaultdict(list)

    for result in results:
        for key, value in result.metrics.retrieval.items():
            retrieval_bucket[key].append(value)
        for key, value in result.metrics.ragas.items():
            ragas_bucket[key].append(value)

    aggregates: dict[str, Any] = {
        "examples": len(results),
        "similarity_avg": fmean(similarity_scores) if similarity_scores else 0.0,
        "processing_time_ms_avg": fmean([item.processing_time_ms for item in results]),
    }

    aggregates["retrieval_avg"] = {
        key: fmean(values) if values else 0.0
        for key, values in retrieval_bucket.items()
    }
    aggregates["ragas_avg"] = {
        key: fmean(values) if values else 0.0 for key, values in ragas_bucket.items()
    }
    return aggregates


async def _evaluate_examples(
    examples: Sequence[GoldenExample],
    orchestrator: SupportsSearchOrchestrator,
    ragas_evaluator: RagasEvaluator,
    *,
    limit: int,
    ragas_sample_cap: int | None = None,
) -> list[ExampleResult]:
    """Execute the orchestrator for every example and collect metrics."""

    # pylint: disable=too-many-locals  # named intermediates keep reporting explicit
    results: list[ExampleResult] = []

    for example in examples:
        request_overrides: dict[str, Any] = {
            "enable_rag": True,
            "limit": limit,
        }
        if collection := example.metadata.get("collection"):
            request_overrides["collection"] = collection

        request_payload = {"query": example.query, **request_overrides}
        request = SearchRequest(**request_payload)
        response: SearchResponse = await orchestrator.search(request)

        predicted = response.generated_answer or " ".join(
            record.content for record in response.records
        )
        contexts = [record.content for record in response.records]

        similarity = _compute_similarity(predicted, example.expected_answer)
        retrieval_metrics = _compute_retrieval_metrics(
            example, [record.model_dump() for record in response.records], k=limit
        )

        if ragas_sample_cap is not None and len(results) >= ragas_sample_cap:
            ragas_scores = {}
        else:
            ragas_scores = ragas_evaluator.evaluate(example, predicted, contexts)

        result = ExampleResult(
            example=example,
            predicted_answer=predicted,
            answer_confidence=response.answer_confidence,
            answer_sources=response.answer_sources or [],
            processing_time_ms=response.processing_time_ms,
            metrics=ExampleMetrics(
                similarity=similarity,
                retrieval=retrieval_metrics,
                ragas=ragas_scores,
            ),
            records=[record.model_dump() for record in response.records],
        )
        results.append(result)

    return results


def _render_report(report: EvaluationReport) -> dict[str, Any]:
    """Convert the dataclass-based report into serialisable structures."""

    return {
        "aggregates": report.aggregates,
        "results": [
            {
                "example": {
                    "query": item.example.query,
                    "expected_answer": item.example.expected_answer,
                    "expected_contexts": item.example.expected_contexts,
                    "references": item.example.references,
                    "metadata": item.example.metadata,
                },
                "predicted_answer": item.predicted_answer,
                "answer_confidence": item.answer_confidence,
                "answer_sources": item.answer_sources,
                "processing_time_ms": item.processing_time_ms,
                "metrics": {
                    "similarity": item.metrics.similarity,
                    "retrieval": item.metrics.retrieval,
                    "ragas": item.metrics.ragas,
                },
                "records": item.records,
            }
            for item in report.results
        ],
    }


def _write_output(payload: dict[str, Any], output_path: Path | None) -> None:
    """Persist the report to disk or pretty-print to stdout."""

    if output_path is None:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Report written to %s", output_path)


async def _run(args: argparse.Namespace) -> None:
    """Execute the evaluation harness with provided CLI arguments."""
    # pylint: disable=too-many-locals, too-many-statements
    dataset_path = Path(args.dataset)
    examples = _load_dataset(dataset_path)

    default_max_samples, _ = _load_cost_controls()
    ragas_sample_cap = args.ragas_max_samples
    if ragas_sample_cap is None:
        ragas_sample_cap = default_max_samples

    ragas_evaluator = RagasEvaluator(
        enabled=args.enable_ragas,
        llm_model=args.ragas_model,
        embedding_model=args.ragas_embedding,
    )
    if args.enable_ragas:
        LOGGER.info(
            "Semantic evaluation enabled. API usage limits should be configured."
        )

    orchestrator = await _load_orchestrator()
    try:
        results = await _evaluate_examples(
            examples,
            orchestrator,
            ragas_evaluator,
            limit=args.limit,
            ragas_sample_cap=ragas_sample_cap,
        )
    finally:
        await orchestrator.cleanup()
        await shutdown_container()

    aggregates = _aggregate_metrics(results)
    thresholds = _load_thresholds()
    gating_failures = _enforce_thresholds(aggregates, thresholds)
    if gating_failures:
        for failure in gating_failures:
            LOGGER.error("Evaluation budget failure: %s", failure)
    report = EvaluationReport(
        results=results,
        aggregates=aggregates,
    )

    payload = _render_report(report)
    _write_output(payload, Path(args.output) if args.output else None)

    if gating_failures:
        raise SystemExit(1)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default="tests/data/rag/golden_set.jsonl",
        help="Path to the golden dataset (JSONL format)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the evaluation report as JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of documents to request per query",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="ml_app",
        help="Metrics namespace for the Prometheus registry",
    )
    parser.add_argument(
        "--enable-ragas",
        action="store_true",
        help="Enable semantic RAGAS metrics (requires configured LLM + embeddings)",
    )
    parser.add_argument(
        "--ragas-model",
        type=str,
        default=None,
        help="LLM model identifier used by RAGAS (defaults to gpt-4o-mini)",
    )
    parser.add_argument(
        "--ragas-embedding",
        type=str,
        default=None,
        help="Embedding model for RAGAS (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--ragas-max-samples",
        type=int,
        default=None,
        help="Upper bound on examples processed with RAGAS (cost control)",
    )
    return parser


def main() -> None:
    """CLI entrypoint for the regression evaluation harness."""

    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
