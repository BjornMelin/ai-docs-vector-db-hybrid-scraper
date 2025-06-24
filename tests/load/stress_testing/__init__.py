"""Stress testing package for AI Documentation Vector DB.

This package provides comprehensive stress testing capabilities including:
- Resource exhaustion testing
- Breaking point identification
- Circuit breaker and rate limiter validation
- System monitoring during stress conditions
- Chaos engineering scenarios

The stress testing suite is designed to push the system beyond normal capacity
to identify failure modes, breaking points, and ensure graceful degradation.
"""

__version__ = "1.0.0"

# Test markers for categorization
STRESS_TEST_MARKERS = [
    "stress",
    "resource_exhaustion",
    "breaking_point",
    "chaos",
    "recovery",
    "slow",
]

# Available stress test profiles
AVAILABLE_PROFILES = [
    "light_stress",
    "moderate_stress",
    "heavy_stress",
    "extreme_stress",
]

# Available chaos scenarios
AVAILABLE_CHAOS_SCENARIOS = [
    "memory_pressure",
    "cpu_spike",
    "network_failure",
    "disk_io_stress",
]
