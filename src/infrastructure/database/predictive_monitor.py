"""Predictive load monitoring with ML-based pattern detection.

This module provides advanced load prediction capabilities using machine learning
to anticipate database load patterns and enable proactive connection pool scaling.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .load_monitor import LoadMetrics
from .load_monitor import LoadMonitor
from .load_monitor import LoadMonitorConfig

logger = logging.getLogger(__name__)


@dataclass
class LoadPrediction:
    """Load prediction result with confidence metrics."""

    predicted_load: float
    confidence_score: float
    recommendation: str
    time_horizon_minutes: int
    feature_importance: Dict[str, float]
    trend_direction: str  # "increasing", "decreasing", "stable"


@dataclass
class PatternFeatures:
    """Extracted features for pattern recognition."""

    avg_requests: float
    peak_requests: float
    memory_trend: float
    response_time_variance: float
    error_rate: float
    time_of_day: float
    day_of_week: float
    cyclical_pattern: float
    volatility_index: float


class PredictiveLoadMonitor(LoadMonitor):
    """Enhanced load monitor with ML-based prediction capabilities.
    
    This class extends the base LoadMonitor with advanced prediction algorithms
    to anticipate future load patterns and recommend proactive scaling actions.
    """

    def __init__(self, config: LoadMonitorConfig):
        """Initialize predictive load monitor.
        
        Args:
            config: Load monitor configuration
        """
        super().__init__(config)
        
        # ML models for prediction
        self.prediction_model = RandomForestRegressor(
            n_estimators=50, random_state=42, max_depth=10
        )
        self.trend_model = LinearRegression()
        self.scaler = StandardScaler()
        
        # Pattern detection
        self.pattern_history: List[LoadMetrics] = []
        self.feature_history: List[PatternFeatures] = []
        self.prediction_cache: Dict[int, LoadPrediction] = {}
        
        # Model state
        self.is_model_trained = False
        self.last_training_time = 0.0
        self.training_interval = 300.0  # Retrain every 5 minutes
        self.min_samples_for_training = 50
        
        # Performance tracking
        self.prediction_accuracy_history: List[float] = []
        self.feature_importance_cache: Dict[str, float] = {}

    async def predict_future_load(
        self, horizon_minutes: int = 15
    ) -> LoadPrediction:
        """Predict database load for specified time horizon.
        
        Args:
            horizon_minutes: Time horizon for prediction in minutes
            
        Returns:
            LoadPrediction with confidence metrics and recommendations
        """
        # Check cache first
        cache_key = horizon_minutes
        if cache_key in self.prediction_cache:
            cached_prediction = self.prediction_cache[cache_key]
            # Use cached prediction if it's less than 1 minute old
            if time.time() - cached_prediction.time_horizon_minutes < 60:
                return cached_prediction

        if not self.is_model_trained or len(self.pattern_history) < 10:
            return await self._bootstrap_prediction(horizon_minutes)

        try:
            # Extract features from recent patterns
            features = self._extract_features_for_prediction()
            
            # Scale features
            features_scaled = self.scaler.transform([features])
            
            # Generate prediction
            predicted_load = self.prediction_model.predict(features_scaled)[0]
            predicted_load = max(0.0, min(1.0, predicted_load))  # Clamp to valid range
            
            # Calculate confidence based on model performance
            confidence = self._calculate_prediction_confidence(features)
            
            # Determine trend direction
            trend_direction = self._analyze_trend_direction()
            
            # Generate recommendation
            recommendation = self._generate_scaling_recommendation(
                predicted_load, confidence, trend_direction
            )
            
            # Get feature importance
            feature_importance = self._get_feature_importance()
            
            prediction = LoadPrediction(
                predicted_load=predicted_load,
                confidence_score=confidence,
                recommendation=recommendation,
                time_horizon_minutes=horizon_minutes,
                feature_importance=feature_importance,
                trend_direction=trend_direction,
            )
            
            # Cache the prediction
            self.prediction_cache[cache_key] = prediction
            
            logger.debug(
                f"Predicted load: {predicted_load:.3f}, "
                f"confidence: {confidence:.3f}, "
                f"trend: {trend_direction}"
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Failed to generate load prediction: {e}")
            return await self._bootstrap_prediction(horizon_minutes)

    async def train_prediction_model(self) -> bool:
        """Train ML model on historical load data.
        
        Returns:
            True if training was successful, False otherwise
        """
        current_time = time.time()
        
        # Check if we should retrain
        if (
            self.is_model_trained
            and current_time - self.last_training_time < self.training_interval
        ):
            return True
            
        if len(self.pattern_history) < self.min_samples_for_training:
            logger.debug(
                f"Insufficient data for training: {len(self.pattern_history)} "
                f"< {self.min_samples_for_training}"
            )
            return False

        try:
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if len(X) < 10:  # Need minimum samples
                return False
                
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train prediction model
            self.prediction_model.fit(X_scaled, y)
            
            # Train trend model for simple linear trends
            time_features = np.arange(len(y)).reshape(-1, 1)
            self.trend_model.fit(time_features, y)
            
            # Update model state
            self.is_model_trained = True
            self.last_training_time = current_time
            
            # Update feature importance cache
            if hasattr(self.prediction_model, 'feature_importances_'):
                feature_names = self._get_feature_names()
                self.feature_importance_cache = dict(
                    zip(feature_names, self.prediction_model.feature_importances_)
                )
            
            # Validate model performance
            await self._validate_model_performance(X_scaled, y)
            
            logger.info(
                f"Successfully trained prediction model with {len(X)} samples"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to train prediction model: {e}")
            return False

    async def record_load_metrics(self, metrics: LoadMetrics) -> None:
        """Record load metrics and update pattern history.
        
        Args:
            metrics: Load metrics to record
        """
        # Call parent method
        await super().record_load_metrics(metrics)
        
        # Add to pattern history
        self.pattern_history.append(metrics)
        
        # Extract and store features
        if len(self.pattern_history) >= 10:
            features = self._extract_pattern_features(metrics)
            self.feature_history.append(features)
        
        # Maintain history size
        max_history = 1000
        if len(self.pattern_history) > max_history:
            self.pattern_history = self.pattern_history[-max_history:]
            self.feature_history = self.feature_history[-max_history:]
        
        # Trigger retraining if needed
        if len(self.pattern_history) % 20 == 0:  # Every 20 metrics
            await self.train_prediction_model()
        
        # Clear old predictions from cache
        self.prediction_cache.clear()

    def _extract_features_for_prediction(self) -> List[float]:
        """Extract features from recent metrics for prediction."""
        if len(self.pattern_history) < 10:
            return [0.0] * 9  # Return zero features
            
        recent_metrics = self.pattern_history[-30:]  # Last 30 metrics
        
        # Calculate feature values
        requests = [m.concurrent_requests for m in recent_metrics]
        memory_usage = [m.memory_usage_percent for m in recent_metrics]
        response_times = [m.avg_response_time_ms for m in recent_metrics]
        errors = [m.connection_errors for m in recent_metrics]
        
        current_time = time.time()
        time_of_day = (current_time % 86400) / 86400  # Normalize to 0-1
        day_of_week = ((current_time // 86400) % 7) / 7  # Normalize to 0-1
        
        features = [
            np.mean(requests),  # avg_requests
            np.max(requests),   # peak_requests
            self._calculate_trend(memory_usage),  # memory_trend
            np.var(response_times) if response_times else 0.0,  # response_time_variance
            np.mean(errors) if errors else 0.0,  # error_rate
            time_of_day,        # time_of_day
            day_of_week,        # day_of_week
            self._detect_cyclical_pattern(recent_metrics),  # cyclical_pattern
            self._calculate_volatility_index(recent_metrics),  # volatility_index
        ]
        
        return features

    def _extract_pattern_features(self, metrics: LoadMetrics) -> PatternFeatures:
        """Extract pattern features from current metrics."""
        if len(self.pattern_history) < 10:
            return PatternFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0)
            
        recent_metrics = self.pattern_history[-10:]
        
        requests = [m.concurrent_requests for m in recent_metrics]
        memory_usage = [m.memory_usage_percent for m in recent_metrics]
        response_times = [m.avg_response_time_ms for m in recent_metrics]
        
        current_time = time.time()
        
        return PatternFeatures(
            avg_requests=np.mean(requests),
            peak_requests=np.max(requests),
            memory_trend=self._calculate_trend(memory_usage),
            response_time_variance=np.var(response_times) if response_times else 0.0,
            error_rate=metrics.connection_errors,
            time_of_day=(current_time % 86400) / 86400,
            day_of_week=((current_time // 86400) % 7) / 7,
            cyclical_pattern=self._detect_cyclical_pattern(recent_metrics),
            volatility_index=self._calculate_volatility_index(recent_metrics),
        )

    def _prepare_training_data(self) -> tuple[List[List[float]], List[float]]:
        """Prepare training data from historical patterns."""
        X = []
        y = []
        
        # Need enough history for windowing
        if len(self.pattern_history) < 30:
            return X, y
            
        # Create training samples with sliding windows
        window_size = 10
        prediction_offset = 5  # Predict 5 steps ahead
        
        for i in range(window_size, len(self.pattern_history) - prediction_offset):
            # Features from current window
            window_metrics = self.pattern_history[i-window_size:i]
            features = self._extract_features_from_window(window_metrics)
            
            # Target is load factor 5 steps ahead
            future_metrics = self.pattern_history[i + prediction_offset]
            target_load = self.calculate_load_factor(future_metrics)
            
            X.append(features)
            y.append(target_load)
            
        return X, y

    def _extract_features_from_window(self, metrics_window: List[LoadMetrics]) -> List[float]:
        """Extract features from a window of metrics."""
        if not metrics_window:
            return [0.0] * 9
            
        requests = [m.concurrent_requests for m in metrics_window]
        memory_usage = [m.memory_usage_percent for m in metrics_window]
        response_times = [m.avg_response_time_ms for m in metrics_window]
        errors = [m.connection_errors for m in metrics_window]
        
        # Use timestamp from last metric for time features
        current_time = metrics_window[-1].timestamp
        time_of_day = (current_time % 86400) / 86400
        day_of_week = ((current_time // 86400) % 7) / 7
        
        return [
            np.mean(requests),
            np.max(requests),
            self._calculate_trend(memory_usage),
            np.var(response_times) if response_times else 0.0,
            np.mean(errors) if errors else 0.0,
            time_of_day,
            day_of_week,
            self._detect_cyclical_pattern(metrics_window),
            self._calculate_volatility_index(metrics_window),
        ]

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction from a series of values."""
        if len(values) < 2:
            return 0.0
            
        # Simple linear regression slope
        x = np.arange(len(values))
        y = np.array(values)
        
        if np.var(x) == 0:
            return 0.0
            
        slope = np.cov(x, y)[0, 1] / np.var(x)
        return float(slope)

    def _detect_cyclical_pattern(self, metrics: List[LoadMetrics]) -> float:
        """Detect cyclical patterns in metrics."""
        if len(metrics) < 10:
            return 0.0
            
        # Simple autocorrelation at different lags
        values = [m.concurrent_requests for m in metrics]
        
        if len(values) < 5:
            return 0.0
            
        # Calculate autocorrelation at lag 1
        mean_val = np.mean(values)
        numerator = sum((values[i] - mean_val) * (values[i-1] - mean_val) 
                       for i in range(1, len(values)))
        denominator = sum((v - mean_val) ** 2 for v in values)
        
        if denominator == 0:
            return 0.0
            
        autocorr = numerator / denominator
        return float(autocorr)

    def _calculate_volatility_index(self, metrics: List[LoadMetrics]) -> float:
        """Calculate volatility index from metrics."""
        if len(metrics) < 2:
            return 0.0
            
        values = [m.concurrent_requests for m in metrics]
        if not values:
            return 0.0
            
        # Coefficient of variation
        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0
            
        cv = np.std(values) / mean_val
        return float(cv)

    def _analyze_trend_direction(self) -> str:
        """Analyze overall trend direction."""
        if len(self.pattern_history) < 10:
            return "stable"
            
        recent_loads = [
            self.calculate_load_factor(m) for m in self.pattern_history[-10:]
        ]
        
        trend = self._calculate_trend(recent_loads)
        
        if trend > 0.01:
            return "increasing"
        elif trend < -0.01:
            return "decreasing"
        else:
            return "stable"

    def _generate_scaling_recommendation(
        self, predicted_load: float, confidence: float, trend: str
    ) -> str:
        """Generate scaling recommendation based on prediction."""
        high_load_threshold = 0.8
        medium_load_threshold = 0.6
        low_confidence_threshold = 0.5
        
        if confidence < low_confidence_threshold:
            return "Monitor closely - prediction confidence is low"
            
        if predicted_load > high_load_threshold:
            if trend == "increasing":
                return "Scale up immediately - high load predicted with increasing trend"
            else:
                return "Prepare to scale up - high load predicted"
        elif predicted_load > medium_load_threshold:
            if trend == "increasing":
                return "Consider scaling up - moderate load with increasing trend"
            else:
                return "Monitor - moderate load predicted"
        else:
            if trend == "decreasing" and predicted_load < 0.3:
                return "Consider scaling down - low load with decreasing trend"
            else:
                return "Maintain current capacity - low to moderate load predicted"

    def _calculate_prediction_confidence(self, features: List[float]) -> float:
        """Calculate confidence score for prediction."""
        base_confidence = 0.7
        
        # Adjust based on training data size
        if len(self.pattern_history) > 100:
            data_confidence = 0.9
        elif len(self.pattern_history) > 50:
            data_confidence = 0.8
        else:
            data_confidence = 0.6
            
        # Adjust based on recent prediction accuracy
        accuracy_confidence = 0.7
        if self.prediction_accuracy_history:
            recent_accuracy = np.mean(self.prediction_accuracy_history[-10:])
            accuracy_confidence = min(0.95, max(0.5, recent_accuracy))
            
        # Combined confidence
        confidence = (base_confidence + data_confidence + accuracy_confidence) / 3
        return float(confidence)

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.feature_importance_cache:
            feature_names = self._get_feature_names()
            return {name: 0.0 for name in feature_names}
            
        return self.feature_importance_cache.copy()

    def _get_feature_names(self) -> List[str]:
        """Get feature names for importance mapping."""
        return [
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

    async def _bootstrap_prediction(self, horizon_minutes: int) -> LoadPrediction:
        """Generate bootstrap prediction when model is not ready."""
        # Simple heuristic-based prediction
        if len(self.pattern_history) > 0:
            recent_metrics = self.pattern_history[-5:]
            avg_load = np.mean([self.calculate_load_factor(m) for m in recent_metrics])
            predicted_load = min(1.0, max(0.0, avg_load))
        else:
            predicted_load = 0.5  # Default moderate load
            
        return LoadPrediction(
            predicted_load=predicted_load,
            confidence_score=0.5,  # Low confidence for bootstrap
            recommendation="Insufficient data for accurate prediction - monitor patterns",
            time_horizon_minutes=horizon_minutes,
            feature_importance={},
            trend_direction="unknown",
        )

    async def _validate_model_performance(
        self, X_scaled: np.ndarray, y: np.ndarray
    ) -> None:
        """Validate model performance and update accuracy tracking."""
        if len(X_scaled) < 10:
            return
            
        try:
            # Simple validation using last 20% of data
            split_idx = int(len(X_scaled) * 0.8)
            X_val = X_scaled[split_idx:]
            y_val = y[split_idx:]
            
            predictions = self.prediction_model.predict(X_val)
            
            # Calculate accuracy metrics
            mae = np.mean(np.abs(predictions - y_val))
            mse = np.mean((predictions - y_val) ** 2)
            
            # Convert to accuracy score (1 - normalized error)
            max_error = 1.0  # Since load factors are 0-1
            accuracy = max(0.0, 1.0 - (mae / max_error))
            
            self.prediction_accuracy_history.append(accuracy)
            
            # Keep only recent accuracy history
            if len(self.prediction_accuracy_history) > 50:
                self.prediction_accuracy_history = self.prediction_accuracy_history[-50:]
                
            logger.debug(
                f"Model validation: MAE={mae:.3f}, MSE={mse:.3f}, Accuracy={accuracy:.3f}"
            )
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")

    async def get_prediction_metrics(self) -> Dict[str, Any]:
        """Get comprehensive prediction performance metrics."""
        return {
            "model_trained": self.is_model_trained,
            "training_samples": len(self.pattern_history),
            "last_training_time": self.last_training_time,
            "prediction_accuracy_avg": (
                np.mean(self.prediction_accuracy_history)
                if self.prediction_accuracy_history
                else 0.0
            ),
            "feature_importance": self.feature_importance_cache,
            "cache_size": len(self.prediction_cache),
        }