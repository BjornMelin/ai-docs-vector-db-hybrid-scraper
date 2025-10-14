"""Tests for observability FastAPI dependencies."""

from unittest.mock import patch

from src.services.observability import dependencies


class TestObservabilityService:
    """Tests for the observability service dependencies."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        dependencies.get_observability_service.cache_clear()

    def test_service_initialises_observability(self) -> None:
        """Test that the observability service initializes correctly."""
        with (
            patch(
                "src.services.observability.dependencies.initialize_observability",
                return_value=True,
            ) as mock_init,
            patch(
                "src.services.observability.dependencies.get_tracer",
                return_value="tracer",
            ) as mock_tracer,
            patch(
                "src.services.observability.dependencies.get_ai_tracker",
                return_value="tracker",
            ),
        ):
            service = dependencies.get_observability_service()

        assert service["enabled"] is True
        assert service["tracer"] == "tracer"
        assert service["ai_tracker"] == "tracker"
        mock_init.assert_called_once()
        mock_tracer.assert_called_once()

    async def test_record_ai_operation_metrics(self) -> None:
        """Test recording AI operation metrics."""
        service = {"enabled": True}
        with patch(
            "src.services.observability.dependencies.record_ai_operation"
        ) as record:  # type: ignore[attr-defined]
            await dependencies.record_ai_operation_metrics(
                "llm",
                "openai",
                "gpt",
                duration_s=0.1,
                tokens=5,
                cost_usd=0.01,
                service=service,
            )
        record.assert_called_once()

    async def test_track_ai_cost_metrics_disabled(self) -> None:
        """Test tracking AI cost metrics when service is disabled."""
        service = {"enabled": False}
        with patch("src.services.observability.dependencies.track_cost") as track_cost:
            await dependencies.track_ai_cost_metrics(
                "llm",
                "openai",
                "gpt",
                cost_usd=0.01,
                service=service,
            )
        track_cost.assert_not_called()
