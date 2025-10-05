"""Tests for the telemetry repository."""

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "src/services/monitoring/telemetry_repository.py"
)

for module_name in (
    "src.config",
    "src.services",
    "src.services.monitoring.telemetry_repository",
):
    sys.modules.pop(module_name, None)

_spec = spec_from_file_location("_telemetry_repository_under_test", MODULE_PATH)
assert _spec and _spec.loader
_module = module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)  # type: ignore[arg-type]

TelemetryRepository = _module.TelemetryRepository
get_telemetry_repository = _module.get_telemetry_repository


def test_increment_counter_records_values() -> None:
    """Counters should accumulate per tag combination."""

    repository = TelemetryRepository()
    repository.increment_counter("runs")
    repository.increment_counter("runs", value=2, tags={"mode": "search"})
    repository.increment_counter("runs", tags={"mode": "search"})

    samples = list(repository.counter_samples("runs"))
    tags_to_value = {sample.tags: sample.value for sample in samples}
    assert tags_to_value.get(()) == 1
    assert tags_to_value.get((("mode", "search"),)) == 3


def test_record_observation_tracks_histogram() -> None:
    """Histogram observations should capture counts and sums."""

    repository = TelemetryRepository()
    repository.record_observation("latency", 5.0)
    repository.record_observation("latency", 3.0, tags={"stage": "retrieval"})
    repository.record_observation("latency", 2.0, tags={"stage": "retrieval"})

    default_samples = [
        sample for sample in repository.histogram_samples("latency") if not sample.tags
    ]
    assert len(default_samples) == 1
    assert default_samples[0].count == 1
    assert default_samples[0].sum == 5.0

    tagged_samples = [
        sample for sample in repository.histogram_samples("latency") if sample.tags
    ]
    assert len(tagged_samples) == 1
    assert tagged_samples[0].count == 2
    assert tagged_samples[0].sum == 5.0


def test_singleton_repository_reset() -> None:
    """Singleton repository should be reusable across calls."""

    repository = get_telemetry_repository()
    repository.reset()
    repository.increment_counter("runs")
    assert list(repository.counter_samples("runs"))
    repository.reset()
    assert not list(repository.counter_samples("runs"))
