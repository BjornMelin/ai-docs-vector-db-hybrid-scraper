"""Shared utilities for Qdrant vector database operations."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from qdrant_client import models


def build_filter(filters: dict[str, Any] | None) -> models.Filter | None:
    """Translate a flexible filter dictionary into a Qdrant ``Filter``.

    The helper understands simple equality/range filters, composite boolean logic,
    relative temporal windows, and ``MatchAny``/``MatchExcept`` expressions. It is
    shared by MCP tools and background services so that all query paths rely on
    the same Qdrant-native filtering primitives.
    """

    if not filters:
        return None

    filter_obj = _build_filter(filters)
    if filter_obj is None:
        return None
    if isinstance(filter_obj, models.Condition):
        return models.Filter(must=[filter_obj])
    return filter_obj


def _build_filter(filters: Any) -> models.Filter | models.Condition | None:
    """Internal recursive builder supporting nested boolean logic."""

    if filters is None:
        return None

    if isinstance(filters, models.Filter | models.Condition):
        return filters

    if isinstance(filters, list):
        conditions = [_build_filter(item) for item in filters]
        flattened = [cond for cond in conditions if cond is not None]
        if not flattened:
            return None
        must_conditions = [
            cond for cond in flattened if isinstance(cond, models.Condition)
        ]
        nested_filters = [cond for cond in flattened if isinstance(cond, models.Filter)]
        if nested_filters and must_conditions:
            combined = [*must_conditions]
            for fil in nested_filters:
                combined.extend(_conditions_from_filter(fil))
            return models.Filter(must=combined)
        if nested_filters:
            # Merge filters of the same type conservatively as ``must`` clauses
            combined = []
            for fil in nested_filters:
                combined.extend(_conditions_from_filter(fil))
            return models.Filter(must=combined)
        return models.Filter(must=must_conditions)

    if not isinstance(filters, dict):
        msg = "Filter definition must be a dictionary, list, or qdrant model"
        raise TypeError(msg)

    if any(key in filters for key in ("must", "should", "must_not")):
        return _build_boolean_filter(filters)

    conditions = list(_build_conditions_from_mapping(filters).values())
    # ``_build_conditions_from_mapping`` returns dict to detect duplicates; we only
    # need the values here.
    if not conditions:
        return None
    return models.Filter(must=list(conditions))


def _build_boolean_filter(definition: dict[str, Any]) -> models.Filter | None:
    """Construct ``Filter`` with ``must/should/must_not`` structure."""

    must = _collect_conditions(definition.get("must"))
    should = _collect_conditions(definition.get("should"))
    must_not = _collect_conditions(definition.get("must_not"))

    if not any((must, should, must_not)):
        return None

    return models.Filter(
        must=must or None, should=should or None, must_not=must_not or None
    )


def _collect_conditions(items: Any) -> list[models.Condition] | None:
    if not items:
        return None
    if not isinstance(items, Iterable) or isinstance(items, (str, bytes, dict)):
        items = [items]

    collected: list[models.Condition] = []
    for item in items:
        built = _build_filter(item)
        if isinstance(built, models.Filter):
            collected.extend(_conditions_from_filter(built))
        elif isinstance(built, models.Condition):
            collected.append(built)
        elif built is not None:
            msg = f"Unsupported nested filter type: {type(built)!r}"
            raise TypeError(msg)
    return collected or None


def _conditions_from_filter(filter_obj: models.Filter) -> list[models.Condition]:
    conditions: list[models.Condition] = []
    for attr in (filter_obj.must, filter_obj.should, filter_obj.must_not):
        if attr:
            if isinstance(attr, list):
                conditions.extend(attr)
            elif isinstance(attr, models.Condition):
                conditions.append(attr)
    return conditions


def _build_conditions_from_mapping(
    filters: dict[str, Any],
) -> dict[str, models.Condition]:
    """Create simple field conditions while deduplicating by key/operator combo."""

    conditions: dict[str, models.Condition] = {}

    # Keyword equality filters
    for field in [
        "doc_type",
        "language",
        "framework",
        "version",
        "crawl_source",
        "site_name",
        "embedding_model",
        "embedding_provider",
        "url",
        "content_type",
    ]:
        if field in filters:
            value = filters[field]
            if isinstance(value, dict):
                condition = _build_match_condition(field, value)
            elif isinstance(value, (list, tuple, set)):
                condition = models.FieldCondition(
                    key=field, match=models.MatchAny(any=list(value))
                )
            else:
                condition = models.FieldCondition(
                    key=field, match=models.MatchValue(value=value)
                )
            conditions[f"eq:{field}"] = condition

    # Text contains filters
    for field in ["title", "content_preview"]:
        if field in filters:
            value = filters[field]
            if isinstance(value, dict):
                conditions[f"text:{field}"] = _build_match_condition(field, value)
            elif isinstance(value, str):
                conditions[f"text:{field}"] = models.FieldCondition(
                    key=field, match=models.MatchText(text=value)
                )
            else:
                msg = f"Text filter value for {field} must be a string"
                raise ValueError(msg)

    # Range filters for timestamps and metrics
    for filter_key, field_key, operator in [
        ("created_after", "created_at", "gte"),
        ("created_before", "created_at", "lte"),
        ("updated_after", "last_updated", "gte"),
        ("updated_before", "last_updated", "lte"),
        ("crawled_after", "crawl_timestamp", "gte"),
        ("crawled_before", "crawl_timestamp", "lte"),
        ("min_word_count", "word_count", "gte"),
        ("max_word_count", "word_count", "lte"),
        ("min_char_count", "char_count", "gte"),
        ("max_char_count", "char_count", "lte"),
        ("min_quality_score", "quality_score", "gte"),
        ("max_quality_score", "quality_score", "lte"),
        ("min_score", "score", "gte"),
        ("min_total_chunks", "total_chunks", "gte"),
        ("max_total_chunks", "total_chunks", "lte"),
        ("min_links_count", "links_count", "gte"),
        ("max_links_count", "links_count", "lte"),
    ]:
        if filter_key in filters:
            range_params = {operator: filters[filter_key]}
            conditions[f"range:{field_key}:{operator}"] = models.FieldCondition(
                key=field_key, range=models.Range(**range_params)
            )

    # Generic range expressions
    if "range" in filters:
        ranges = filters["range"]
        if not isinstance(ranges, dict):
            msg = "range filter must be a dictionary"
            raise TypeError(msg)
        for field, params in ranges.items():
            if not isinstance(params, dict):
                msg = f"Range parameters for {field} must be a dictionary"
                raise TypeError(msg)
            conditions[f"range:{field}"] = models.FieldCondition(
                key=field, range=models.Range(**params)
            )

    # Explicit equality for structural fields
    for field in ["chunk_index", "depth"]:
        if field in filters:
            conditions[f"eq:{field}"] = models.FieldCondition(
                key=field, match=models.MatchValue(value=filters[field])
            )

    # Temporal shortcuts
    if "temporal" in filters:
        temporal_config = filters["temporal"]
        if not isinstance(temporal_config, dict):
            msg = "temporal filter must be a dictionary"
            raise TypeError(msg)
        conditions.update(_build_temporal_conditions(temporal_config))

    # Allow arbitrary field definitions using ``field_conditions`` to avoid
    # repeated schema updates.
    if "field_conditions" in filters:
        field_conditions = filters["field_conditions"]
        if not isinstance(field_conditions, Iterable):
            msg = "field_conditions must be an iterable"
            raise TypeError(msg)
        for idx, item in enumerate(field_conditions):
            built = _build_filter(item)
            if isinstance(built, models.Condition):
                conditions[f"custom:{idx}"] = built
            elif isinstance(built, models.Filter):
                for nested_idx, condition in enumerate(_conditions_from_filter(built)):
                    conditions[f"custom:{idx}:{nested_idx}"] = condition

    return conditions


def _build_match_condition(field: str, spec: dict[str, Any]) -> models.FieldCondition:
    if "any" in spec:
        values = spec["any"]
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            msg = f"MatchAny for {field} must be provided as an iterable"
            raise TypeError(msg)
        return models.FieldCondition(key=field, match=models.MatchAny(any=list(values)))

    if "not" in spec:
        values = spec["not"]
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            msg = f"MatchExcept for {field} must be provided as an iterable"
            raise TypeError(msg)
        return models.FieldCondition(
            key=field, match=models.MatchExcept(**{"except": list(values)})
        )

    if "value" in spec:
        return models.FieldCondition(
            key=field, match=models.MatchValue(value=spec["value"])
        )

    if "text" in spec:
        return models.FieldCondition(
            key=field, match=models.MatchText(text=spec["text"])
        )

    msg = f"Unsupported match specification for field {field}: {spec}"
    raise ValueError(msg)


def _build_temporal_conditions(config: dict[str, Any]) -> dict[str, models.Condition]:
    field = config.get("field", "created_at")
    conditions: dict[str, models.Condition] = {}

    start = _parse_datetime(config.get("start")) or _parse_time_window(
        config.get("window")
    )
    end = _parse_datetime(config.get("end"))

    if start:
        conditions[f"temporal:{field}:gte"] = models.FieldCondition(
            key=field, range=models.Range(gte=start.timestamp())
        )

    if end:
        conditions[f"temporal:{field}:lte"] = models.FieldCondition(
            key=field, range=models.Range(lte=end.timestamp())
        )

    if "between" in config:
        between = config["between"]
        if not isinstance(between, dict):
            msg = "temporal 'between' must be a dictionary"
            raise TypeError(msg)
        start_value = _parse_datetime(between.get("start"))
        end_value = _parse_datetime(between.get("end"))
        timestamp_range: dict[str, float] = {}
        if start_value:
            timestamp_range["gte"] = start_value.timestamp()
        if end_value:
            timestamp_range["lte"] = end_value.timestamp()
        if timestamp_range:
            conditions[f"temporal:{field}:between"] = models.FieldCondition(
                key=field, range=models.Range(**timestamp_range)
            )

    return conditions


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _parse_time_window(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if len(normalized) < 2:
        return None
    multiplier_map = {
        "h": 3600,
        "d": 86400,
        "w": 604800,
        "m": 2592000,
    }
    unit = normalized[-1]
    if unit not in multiplier_map:
        return None
    try:
        amount = int(normalized[:-1])
    except ValueError:
        return None
    delta = timedelta(seconds=amount * multiplier_map[unit])
    return datetime.now(tz=UTC) - delta
