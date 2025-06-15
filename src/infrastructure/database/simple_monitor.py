"""Simple load monitoring with threshold-based logic.

This module provides essential load monitoring without ML complexity:
- Basic CPU and memory monitoring
- Simple threshold-based scaling decisions
- Clear, maintainable logic

Philosophy: Use simple thresholds and system metrics rather than ML prediction.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

import psutil

from .load_monitor import LoadMetrics
from .load_monitor import LoadMonitor
from .load_monitor import LoadMonitorConfig

logger = logging.getLogger(__name__)


@dataclass
class SimpleLoadDecision:
    """Simple load decision with clear reasoning."""

    should_scale_up: bool
    should_scale_down: bool
    current_load: float
    reason: str
    recommendation: str


class SimpleLoadMonitor(LoadMonitor):
    """Simple load monitor using basic thresholds and system metrics.

    Replaces the 726-line ML predictor with straightforward logic:
    - Monitor CPU and memory usage
    - Use configurable thresholds for scaling decisions
    - Clear, debuggable logic without ML complexity
    """

    def __init__(self, config: LoadMonitorConfig):
        """Initialize simple load monitor.

        Args:
            config: Load monitor configuration
        """
        super().__init__(config)

        # Simple thresholds (configurable)
        self.scale_up_threshold = 0.8  # Scale up at 80% load
        self.scale_down_threshold = 0.3  # Scale down at 30% load
        self.memory_threshold = 0.85  # Memory warning at 85%

        # Simple state tracking
        self.recent_loads: list[float] = []
        self.last_scale_decision = 0.0
        self.scale_cooldown = 300.0  # 5 minutes between scale decisions

    async def get_current_load(self) -> LoadMetrics:
        """Get current system load metrics.

        Returns:
            LoadMetrics object with current system state
        """
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Simple weighted average: CPU is more important for database load
            load = (cpu_percent * 0.7 + memory_percent * 0.3) / 100.0

            # Clamp to valid range
            load = max(0.0, min(1.0, load))

            # Track recent loads for trend analysis
            self.recent_loads.append(load)
            if len(self.recent_loads) > 10:
                self.recent_loads = self.recent_loads[-10:]

            # Return LoadMetrics for compatibility
            return LoadMetrics(
                concurrent_requests=self._active_requests,
                memory_usage_percent=memory_percent,
                cpu_usage_percent=cpu_percent,
                avg_response_time_ms=self._avg_response_time,
                connection_errors=self._connection_errors,
                timestamp=time.time(),
            )

        except Exception as e:
            logger.error(f"Failed to get current load: {e}")
            # Return default metrics
            return LoadMetrics(
                concurrent_requests=0,
                memory_usage_percent=50.0,
                cpu_usage_percent=50.0,
                avg_response_time_ms=0.0,
                connection_errors=0,
                timestamp=time.time(),
            )

    def _get_simple_load(self) -> float:
        """Get simple load value as a float 0-1.

        Returns:
            Load factor between 0.0 and 1.0
        """
        try:
            # Get CPU usage (already 0-100, convert to 0-1)
            cpu_percent = psutil.cpu_percent(interval=1) / 100.0

            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent / 100.0

            # Simple weighted average: CPU is more important for database load
            load = (cpu_percent * 0.7) + (memory_percent * 0.3)

            # Clamp to valid range
            return max(0.0, min(1.0, load))

        except Exception as e:
            logger.error(f"Failed to get simple load: {e}")
            return 0.5  # Default moderate load

    def should_scale_up(self, current_load: float | None = None) -> bool:
        """Determine if we should scale up based on simple threshold.

        Args:
            current_load: Optional current load (gets current if None)

        Returns:
            True if we should scale up
        """
        if current_load is None:
            current_load = self._get_simple_load()

        # Simple threshold check
        if current_load > self.scale_up_threshold:
            # Check if we're in cooldown period
            if time.time() - self.last_scale_decision > self.scale_cooldown:
                # Check trend - only scale if load is consistently high
                if len(self.recent_loads) >= 3:
                    recent_high = sum(
                        1
                        for load in self.recent_loads[-3:]
                        if load > self.scale_up_threshold
                    )
                    if recent_high >= 2:  # 2 out of 3 recent samples are high
                        return True
                else:
                    return True  # Not enough history, scale up on threshold

        return False

    def should_scale_down(self, current_load: float | None = None) -> bool:
        """Determine if we should scale down based on simple threshold.

        Args:
            current_load: Optional current load (gets current if None)

        Returns:
            True if we should scale down
        """
        if current_load is None:
            current_load = self.get_current_load()

        # Simple threshold check
        if current_load < self.scale_down_threshold:
            # Check if we're in cooldown period
            if time.time() - self.last_scale_decision > self.scale_cooldown:
                # Check trend - only scale down if load is consistently low
                if len(self.recent_loads) >= 5:
                    recent_low = sum(
                        1
                        for load in self.recent_loads[-5:]
                        if load < self.scale_down_threshold
                    )
                    if recent_low >= 4:  # 4 out of 5 recent samples are low
                        return True

        return False

    def get_load_decision(self) -> SimpleLoadDecision:
        """Get load decision with clear reasoning.

        Returns:
            SimpleLoadDecision with recommendation and reasoning
        """
        current_load = self._get_simple_load()
        should_scale_up = self.should_scale_up(current_load)
        should_scale_down = self.should_scale_down(current_load)

        # Generate clear reasoning
        if should_scale_up:
            reason = (
                f"Load {current_load:.1%} > {self.scale_up_threshold:.1%} threshold"
            )
            recommendation = "Scale up database connections"
        elif should_scale_down:
            reason = (
                f"Load {current_load:.1%} < {self.scale_down_threshold:.1%} threshold"
            )
            recommendation = "Scale down database connections"
        else:
            reason = f"Load {current_load:.1%} within normal range"
            recommendation = "Maintain current capacity"

        return SimpleLoadDecision(
            should_scale_up=should_scale_up,
            should_scale_down=should_scale_down,
            current_load=current_load,
            reason=reason,
            recommendation=recommendation,
        )

    def record_scale_decision(self) -> None:
        """Record that a scaling decision was made."""
        self.last_scale_decision = time.time()

    def get_system_stats(self) -> dict[str, Any]:
        """Get basic system statistics for monitoring.

        Returns:
            Dictionary with system stats
        """
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "current_load": self._get_simple_load(),
                "recent_loads": self.recent_loads.copy(),
                "scale_up_threshold": self.scale_up_threshold,
                "scale_down_threshold": self.scale_down_threshold,
            }

        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {"error": str(e)}

    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure.

        Returns:
            True if memory usage is critically high
        """
        try:
            memory = psutil.virtual_memory()
            return memory.percent / 100.0 > self.memory_threshold
        except Exception:
            return False

    def get_load_trend(self) -> str:
        """Get simple load trend analysis.

        Returns:
            Trend description: "increasing", "decreasing", or "stable"
        """
        if len(self.recent_loads) < 3:
            return "stable"

        recent = self.recent_loads[-3:]

        # Simple trend: compare first and last
        if recent[-1] > recent[0] + 0.1:
            return "increasing"
        elif recent[-1] < recent[0] - 0.1:
            return "decreasing"
        else:
            return "stable"
