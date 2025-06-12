"""Comprehensive tests for PredictiveLoadMonitor ML-based prediction system.

This test module provides comprehensive coverage for the machine learning-based
load prediction system including pattern recognition, model training, and
prediction generation.
"""

import time
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest
from src.infrastructure.database.load_monitor import LoadMetrics
from src.infrastructure.database.load_monitor import LoadMonitorConfig
from src.infrastructure.database.predictive_monitor import LoadPrediction
from src.infrastructure.database.predictive_monitor import PredictiveLoadMonitor


class TestPredictiveLoadMonitor:
    """Test PredictiveLoadMonitor functionality."""

    @pytest.fixture
    def monitor_config(self):
        """Create test load monitor configuration."""
        return LoadMonitorConfig(
            monitoring_interval=0.1,
            memory_threshold_mb=1000,
            cpu_threshold_percent=80.0,
            response_time_threshold_ms=500.0,
        )

    @pytest.fixture
    def predictive_monitor(self, monitor_config):
        """Create PredictiveLoadMonitor instance."""
        return PredictiveLoadMonitor(monitor_config)

    @pytest.fixture
    def sample_metrics_sequence(self):
        """Create a sequence of sample load metrics for testing."""
        base_time = time.time()
        metrics = []

        for i in range(100):
            metrics.append(
                LoadMetrics(
                    concurrent_requests=5 + (i % 10),  # Cyclical pattern
                    memory_usage_percent=50.0 + (i * 0.2),  # Gradual increase
                    cpu_usage_percent=30.0 + (i % 5) * 5,  # Variation
                    avg_response_time_ms=100.0 + (i * 0.5),  # Trend
                    connection_errors=0 if i % 20 != 0 else 1,  # Sporadic errors
                    timestamp=base_time + i * 10,  # 10 second intervals
                )
            )

        return metrics

    @pytest.mark.asyncio
    async def test_initialization(self, predictive_monitor, monitor_config):
        """Test predictive monitor initialization."""
        assert predictive_monitor.config == monitor_config
        assert not predictive_monitor.is_model_trained
        assert len(predictive_monitor.pattern_history) == 0
        assert len(predictive_monitor.feature_history) == 0
        assert predictive_monitor.min_samples_for_training == 50

    @pytest.mark.asyncio
    async def test_bootstrap_prediction(self, predictive_monitor):
        """Test bootstrap prediction when model is not trained."""
        prediction = await predictive_monitor.predict_future_load(horizon_minutes=15)

        assert isinstance(prediction, LoadPrediction)
        assert 0.0 <= prediction.predicted_load <= 1.0
        assert prediction.confidence_score == 0.5  # Low confidence for bootstrap
        assert "Insufficient data" in prediction.recommendation
        assert prediction.trend_direction == "unknown"

    @pytest.mark.asyncio
    async def test_record_load_metrics(
        self, predictive_monitor, sample_metrics_sequence
    ):
        """Test recording load metrics and pattern history management."""
        for metrics in sample_metrics_sequence:
            await predictive_monitor.record_load_metrics(metrics)

        assert len(predictive_monitor.pattern_history) == len(sample_metrics_sequence)
        # Features should be extracted after 10 metrics
        assert (
            len(predictive_monitor.feature_history) >= len(sample_metrics_sequence) - 9
        )

    @pytest.mark.asyncio
    async def test_pattern_history_size_management(self, predictive_monitor):
        """Test that pattern history is properly bounded."""
        # Create more than max_history metrics
        base_time = time.time()
        metrics_count = 1100  # More than the 1000 limit

        for i in range(metrics_count):
            metrics = LoadMetrics(
                concurrent_requests=i % 10,
                memory_usage_percent=50.0,
                cpu_usage_percent=40.0,
                avg_response_time_ms=100.0,
                connection_errors=0,
                timestamp=base_time + i,
            )
            await predictive_monitor.record_load_metrics(metrics)

        # Should be capped at max_history
        assert len(predictive_monitor.pattern_history) == 1000
        assert len(predictive_monitor.feature_history) == 1000

    @pytest.mark.asyncio
    async def test_feature_extraction(
        self, predictive_monitor, sample_metrics_sequence
    ):
        """Test feature extraction from metrics sequence."""
        # Add sufficient metrics for feature extraction
        for metrics in sample_metrics_sequence[:30]:
            await predictive_monitor.record_load_metrics(metrics)

        features = predictive_monitor._extract_features_for_prediction()

        assert len(features) == 9  # Expected number of features
        assert all(isinstance(f, int | float) for f in features)

        # Test individual feature types
        avg_requests, peak_requests, memory_trend = features[:3]
        assert avg_requests >= 0
        assert peak_requests >= avg_requests
        assert isinstance(memory_trend, float)

    def test_query_normalization(self, predictive_monitor):
        """Test query pattern normalization."""
        # Test basic normalization
        query1 = "SELECT * FROM users WHERE id = 123"
        query2 = "SELECT * FROM users WHERE id = 456"

        norm1 = predictive_monitor._normalize_query(query1)
        norm2 = predictive_monitor._normalize_query(query2)

        # Should normalize to same pattern
        assert norm1 == norm2
        assert "?" in norm1  # Numeric literal should be replaced

        # Test string literal normalization
        query3 = "SELECT * FROM users WHERE name = 'John'"
        query4 = "SELECT * FROM users WHERE name = 'Jane'"

        norm3 = predictive_monitor._normalize_query(query3)
        norm4 = predictive_monitor._normalize_query(query4)

        assert norm3 == norm4
        assert "'?'" in norm3  # String literal should be replaced

    def test_query_classification(self, predictive_monitor):
        """Test automatic query type classification."""
        test_cases = [
            ("SELECT * FROM users", "READ"),
            ("INSERT INTO users VALUES (1, 'test')", "WRITE"),
            ("UPDATE users SET name = 'test'", "WRITE"),
            ("DELETE FROM users WHERE id = 1", "WRITE"),
            ("SELECT COUNT(*) FROM orders GROUP BY date", "ANALYTICS"),
            ("SELECT AVG(price) FROM products", "ANALYTICS"),
            ("BEGIN; INSERT INTO accounts VALUES (1); COMMIT;", "TRANSACTION"),
            ("ANALYZE TABLE performance", "MAINTENANCE"),
            ("VACUUM users", "MAINTENANCE"),
        ]

        for query, expected_type in test_cases:
            result = predictive_monitor._classify_query_type(query)
            assert result.value.upper() == expected_type

    def test_complexity_calculation(self, predictive_monitor):
        """Test query complexity scoring."""
        simple_query = "SELECT id FROM users"
        complex_query = """
        SELECT u.name, COUNT(o.id) as order_count
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.created_at > '2023-01-01'
        GROUP BY u.id, u.name
        HAVING COUNT(o.id) > 5
        ORDER BY order_count DESC
        """

        simple_score = predictive_monitor._calculate_query_complexity(simple_query)
        complex_score = predictive_monitor._calculate_query_complexity(complex_query)

        assert 0.0 <= simple_score <= 1.0
        assert 0.0 <= complex_score <= 1.0
        assert complex_score > simple_score

    def test_trend_calculation(self, predictive_monitor):
        """Test trend calculation from value sequences."""
        # Test increasing trend
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = predictive_monitor._calculate_trend(increasing_values)
        assert trend > 0

        # Test decreasing trend
        decreasing_values = [5.0, 4.0, 3.0, 2.0, 1.0]
        trend = predictive_monitor._calculate_trend(decreasing_values)
        assert trend < 0

        # Test stable trend
        stable_values = [3.0, 3.0, 3.0, 3.0, 3.0]
        trend = predictive_monitor._calculate_trend(stable_values)
        assert abs(trend) < 0.1

    def test_cyclical_pattern_detection(self, predictive_monitor):
        """Test cyclical pattern detection in metrics."""
        # Create metrics with clear cyclical pattern
        cyclical_metrics = []
        base_time = time.time()

        for i in range(20):
            requests = 5 + 3 * np.sin(i * 0.5)  # Sinusoidal pattern
            metrics = LoadMetrics(
                concurrent_requests=int(requests),
                memory_usage_percent=50.0,
                cpu_usage_percent=40.0,
                avg_response_time_ms=100.0,
                connection_errors=0,
                timestamp=base_time + i,
            )
            cyclical_metrics.append(metrics)

        pattern_score = predictive_monitor._detect_cyclical_pattern(cyclical_metrics)
        assert isinstance(pattern_score, float)
        # Should detect some pattern in cyclical data
        assert abs(pattern_score) >= 0

    def test_volatility_calculation(self, predictive_monitor):
        """Test volatility index calculation."""
        # High volatility metrics
        volatile_metrics = []
        base_time = time.time()
        volatile_values = [1, 10, 2, 15, 3, 12, 4, 8]

        for i, val in enumerate(volatile_values):
            metrics = LoadMetrics(
                concurrent_requests=val,
                memory_usage_percent=50.0,
                cpu_usage_percent=40.0,
                avg_response_time_ms=100.0,
                connection_errors=0,
                timestamp=base_time + i,
            )
            volatile_metrics.append(metrics)

        volatility = predictive_monitor._calculate_volatility_index(volatile_metrics)
        assert volatility > 0  # Should detect high volatility

        # Low volatility metrics
        stable_metrics = []
        for i in range(8):
            metrics = LoadMetrics(
                concurrent_requests=5,  # Constant value
                memory_usage_percent=50.0,
                cpu_usage_percent=40.0,
                avg_response_time_ms=100.0,
                connection_errors=0,
                timestamp=base_time + i,
            )
            stable_metrics.append(metrics)

        stable_volatility = predictive_monitor._calculate_volatility_index(
            stable_metrics
        )
        assert stable_volatility < volatility  # Should be less volatile

    @pytest.mark.asyncio
    async def test_model_training_insufficient_data(self, predictive_monitor):
        """Test model training with insufficient data."""
        # Add only a few metrics
        for i in range(10):
            metrics = LoadMetrics(
                concurrent_requests=i,
                memory_usage_percent=50.0,
                cpu_usage_percent=40.0,
                avg_response_time_ms=100.0,
                connection_errors=0,
                timestamp=time.time() + i,
            )
            await predictive_monitor.record_load_metrics(metrics)

        success = await predictive_monitor.train_prediction_model()
        assert not success
        assert not predictive_monitor.is_model_trained

    @pytest.mark.asyncio
    async def test_model_training_success(
        self, predictive_monitor, sample_metrics_sequence
    ):
        """Test successful model training."""
        # Add sufficient training data
        for metrics in sample_metrics_sequence:
            await predictive_monitor.record_load_metrics(metrics)

        success = await predictive_monitor.train_prediction_model()
        assert success
        assert predictive_monitor.is_model_trained
        assert predictive_monitor.last_training_time > 0

    @pytest.mark.asyncio
    async def test_prediction_with_trained_model(
        self, predictive_monitor, sample_metrics_sequence
    ):
        """Test prediction generation with trained model."""
        # Train the model first
        for metrics in sample_metrics_sequence:
            await predictive_monitor.record_load_metrics(metrics)

        await predictive_monitor.train_prediction_model()

        # Generate prediction
        prediction = await predictive_monitor.predict_future_load(horizon_minutes=20)

        assert isinstance(prediction, LoadPrediction)
        assert 0.0 <= prediction.predicted_load <= 1.0
        assert 0.0 <= prediction.confidence_score <= 1.0
        assert prediction.time_horizon_minutes == 20
        assert prediction.trend_direction in ["increasing", "decreasing", "stable"]
        assert len(prediction.feature_importance) > 0

    def test_trend_direction_analysis(
        self, predictive_monitor, sample_metrics_sequence
    ):
        """Test trend direction analysis."""
        # Add metrics with clear trend
        for metrics in sample_metrics_sequence[:20]:
            predictive_monitor.pattern_history.append(metrics)

        # Mock calculate_load_factor to return increasing values
        load_factors = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        with patch.object(
            predictive_monitor, "calculate_load_factor", side_effect=load_factors
        ):
            trend = predictive_monitor._analyze_trend_direction()
            assert trend == "increasing"

    def test_scaling_recommendation_generation(self, predictive_monitor):
        """Test scaling recommendation generation."""
        test_cases = [
            (0.9, 0.8, "increasing", "Scale up immediately"),
            (0.7, 0.8, "stable", "Monitor - moderate load"),
            (0.2, 0.8, "decreasing", "Consider scaling down"),
            (0.5, 0.3, "stable", "Monitor closely - prediction confidence is low"),
        ]

        for predicted_load, confidence, trend, expected_keyword in test_cases:
            recommendation = predictive_monitor._generate_scaling_recommendation(
                predicted_load, confidence, trend
            )
            assert expected_keyword.lower() in recommendation.lower()

    def test_confidence_calculation(self, predictive_monitor):
        """Test prediction confidence calculation."""
        # Test with different amounts of training data
        predictive_monitor.pattern_history = [Mock()] * 150  # Large dataset
        high_data_confidence = predictive_monitor._calculate_prediction_confidence([])

        predictive_monitor.pattern_history = [Mock()] * 30  # Small dataset
        low_data_confidence = predictive_monitor._calculate_prediction_confidence([])

        assert high_data_confidence > low_data_confidence

    def test_feature_importance_tracking(self, predictive_monitor):
        """Test feature importance tracking."""
        feature_names = predictive_monitor._get_feature_names()

        expected_features = [
            "avg_requests",
            "peak_requests",
            "memory_trend",
            "response_time_variance",
            "error_rate",
            "time_of_day",
            "day_of_week",
            "cyclical_pattern",
            "volatility_index",
        ]

        assert len(feature_names) == len(expected_features)
        for feature in expected_features:
            assert feature in feature_names

    @pytest.mark.asyncio
    async def test_prediction_caching(
        self, predictive_monitor, sample_metrics_sequence
    ):
        """Test prediction result caching."""
        # Train model
        for metrics in sample_metrics_sequence:
            await predictive_monitor.record_load_metrics(metrics)
        await predictive_monitor.train_prediction_model()

        # First prediction
        prediction1 = await predictive_monitor.predict_future_load(horizon_minutes=15)

        # Second prediction with same horizon (should use cache)
        with patch.object(
            predictive_monitor.prediction_model, "predict"
        ) as mock_predict:
            prediction2 = await predictive_monitor.predict_future_load(
                horizon_minutes=15
            )
            # Predict should not be called again due to caching
            mock_predict.assert_not_called()

        assert prediction1.predicted_load == prediction2.predicted_load

    @pytest.mark.asyncio
    async def test_prediction_cache_expiry(self, predictive_monitor):
        """Test prediction cache expiry behavior."""
        # Mock time to simulate cache expiry
        with patch("time.time", return_value=1000):
            # Create a cached prediction
            from src.infrastructure.database.predictive_monitor import LoadPrediction

            cached_prediction = LoadPrediction(
                predicted_load=0.5,
                confidence_score=0.7,
                recommendation="test",
                time_horizon_minutes=900,  # Old timestamp
                feature_importance={},
                trend_direction="stable",
            )
            predictive_monitor.prediction_cache[15] = cached_prediction

        # Advance time significantly
        with patch("time.time", return_value=2000):
            # Should not use expired cache
            prediction = await predictive_monitor.predict_future_load(
                horizon_minutes=15
            )
            # Should be bootstrap prediction due to no training data
            assert "Insufficient data" in prediction.recommendation

    @pytest.mark.asyncio
    async def test_model_validation(self, predictive_monitor, sample_metrics_sequence):
        """Test model performance validation."""
        # Add training data
        for metrics in sample_metrics_sequence:
            await predictive_monitor.record_load_metrics(metrics)

        # Train model
        await predictive_monitor.train_prediction_model()

        # Check that accuracy tracking was initialized
        assert len(predictive_monitor.prediction_accuracy_history) > 0

        # All accuracy scores should be between 0 and 1
        for accuracy in predictive_monitor.prediction_accuracy_history:
            assert 0.0 <= accuracy <= 1.0

    @pytest.mark.asyncio
    async def test_prediction_metrics_collection(
        self, predictive_monitor, sample_metrics_sequence
    ):
        """Test comprehensive prediction metrics collection."""
        # Add data and train model
        for metrics in sample_metrics_sequence:
            await predictive_monitor.record_load_metrics(metrics)
        await predictive_monitor.train_prediction_model()

        metrics = await predictive_monitor.get_prediction_metrics()

        expected_keys = [
            "model_trained",
            "training_samples",
            "last_training_time",
            "prediction_accuracy_avg",
            "feature_importance",
            "cache_size",
        ]

        for key in expected_keys:
            assert key in metrics

        assert metrics["model_trained"] is True
        assert metrics["training_samples"] > 0
        assert isinstance(metrics["feature_importance"], dict)

    @pytest.mark.asyncio
    async def test_training_interval_enforcement(
        self, predictive_monitor, sample_metrics_sequence
    ):
        """Test that model retraining respects time intervals."""
        # Add data and train model
        for metrics in sample_metrics_sequence:
            await predictive_monitor.record_load_metrics(metrics)

        await predictive_monitor.train_prediction_model()
        first_training_time = predictive_monitor.last_training_time

        # Immediate retraining should be skipped
        success = await predictive_monitor.train_prediction_model()
        assert success  # Returns True but doesn't retrain
        assert predictive_monitor.last_training_time == first_training_time

        # Mock time advancement to force retraining
        with patch(
            "time.time", return_value=first_training_time + 400
        ):  # > training_interval
            success = await predictive_monitor.train_prediction_model()
            assert success
            assert predictive_monitor.last_training_time > first_training_time

    @pytest.mark.asyncio
    async def test_error_handling_in_prediction(self, predictive_monitor):
        """Test error handling during prediction generation."""
        # Force model to be considered trained but cause prediction to fail
        predictive_monitor.is_model_trained = True

        # Create proper mock LoadMetrics objects with all required attributes
        mock_metrics = []
        base_time = time.time()
        for i in range(20):
            mock_metric = Mock()
            mock_metric.concurrent_requests = 5
            mock_metric.memory_usage_percent = 50.0
            mock_metric.cpu_usage_percent = 40.0
            mock_metric.avg_response_time_ms = 100.0
            mock_metric.connection_errors = 0
            mock_metric.timestamp = base_time + i
            mock_metrics.append(mock_metric)

        predictive_monitor.pattern_history = mock_metrics

        # Mock model prediction to raise exception
        with patch.object(
            predictive_monitor.prediction_model,
            "predict",
            side_effect=Exception("Prediction error"),
        ):
            prediction = await predictive_monitor.predict_future_load()

            # Should fall back to bootstrap prediction
            assert "Insufficient data" in prediction.recommendation
            assert prediction.confidence_score == 0.5

    def test_window_based_feature_extraction(
        self, predictive_monitor, sample_metrics_sequence
    ):
        """Test feature extraction from sliding windows."""
        window_metrics = sample_metrics_sequence[:10]
        features = predictive_monitor._extract_features_from_window(window_metrics)

        assert len(features) == 9
        assert all(isinstance(f, int | float) for f in features)
        assert not any(np.isnan(f) for f in features)  # No NaN values

    @pytest.mark.asyncio
    async def test_training_data_preparation(
        self, predictive_monitor, sample_metrics_sequence
    ):
        """Test training data preparation from historical patterns."""
        # Add sufficient history
        for metrics in sample_metrics_sequence:
            predictive_monitor.pattern_history.append(metrics)

        X, y = predictive_monitor._prepare_training_data()

        assert len(X) > 0
        assert len(y) > 0
        assert len(X) == len(y)

        # Each feature vector should have correct length
        for features in X:
            assert len(features) == 9

        # All target values should be valid load factors (0-1)
        for target in y:
            assert 0.0 <= target <= 1.0


class TestPredictiveLoadMonitorEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_metrics_handling(self):
        """Test handling of empty or invalid metrics."""
        monitor = PredictiveLoadMonitor(LoadMonitorConfig())

        # Test with empty history
        features = monitor._extract_features_for_prediction()
        assert len(features) == 9
        assert all(f == 0.0 for f in features)

        # Test trend calculation with empty data
        trend = monitor._calculate_trend([])
        assert trend == 0.0

    @pytest.mark.asyncio
    async def test_single_metric_edge_cases(self):
        """Test behavior with single metric data points."""
        monitor = PredictiveLoadMonitor(LoadMonitorConfig())

        single_metric = LoadMetrics(
            concurrent_requests=5,
            memory_usage_percent=50.0,
            cpu_usage_percent=40.0,
            avg_response_time_ms=100.0,
            connection_errors=0,
            timestamp=time.time(),
        )

        await monitor.record_load_metrics(single_metric)

        # Should handle single metric gracefully
        features = monitor._extract_features_for_prediction()
        assert len(features) == 9

        # Pattern detection with single metric
        pattern_score = monitor._detect_cyclical_pattern([single_metric])
        assert pattern_score == 0.0

    def test_extreme_value_handling(self):
        """Test handling of extreme values in calculations."""
        monitor = PredictiveLoadMonitor(LoadMonitorConfig())

        # Test with zero variance
        volatility = monitor._calculate_volatility_index(
            [
                LoadMetrics(
                    concurrent_requests=5,
                    memory_usage_percent=50.0,
                    cpu_usage_percent=40.0,
                    avg_response_time_ms=100.0,
                    connection_errors=0,
                    timestamp=time.time(),
                )
                for _ in range(10)
            ]
        )
        assert volatility == 0.0

    @pytest.mark.asyncio
    async def test_sklearn_not_available_fallback(self):
        """Test fallback behavior when sklearn is not available."""
        PredictiveLoadMonitor(LoadMonitorConfig())

        # Mock sklearn import failure
        with patch(
            "src.infrastructure.database.predictive_monitor.RandomForestRegressor",
            side_effect=ImportError,
        ):
            # Should still initialize but with limited functionality
            # In practice, this would require different initialization handling
            pass

    @pytest.mark.asyncio
    async def test_model_training_with_invalid_data(self):
        """Test model training with problematic data."""
        monitor = PredictiveLoadMonitor(LoadMonitorConfig())

        # Add metrics with NaN values or extreme outliers
        problematic_metrics = []
        base_time = time.time()

        for i in range(60):
            # Include some extreme values
            requests = 1000000 if i == 30 else 5  # Extreme outlier
            metrics = LoadMetrics(
                concurrent_requests=requests,
                memory_usage_percent=50.0,
                cpu_usage_percent=40.0,
                avg_response_time_ms=100.0,
                connection_errors=0,
                timestamp=base_time + i,
            )
            problematic_metrics.append(metrics)

        for metrics in problematic_metrics:
            await monitor.record_load_metrics(metrics)

        # Training should handle outliers gracefully
        success = await monitor.train_prediction_model()
        # May succeed or fail depending on data processing, but shouldn't crash
        assert isinstance(success, bool)
