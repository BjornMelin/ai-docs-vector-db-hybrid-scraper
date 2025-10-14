"""Load test profiles and scenarios for different testing patterns.

This module defines various load profiles including steady, ramp-up, spike,
and step patterns for comprehensive performance testing.
"""

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from locust import LoadTestShape


@dataclass
class LoadStage:
    """Represents a stage in a load test profile."""

    duration: int  # Duration in seconds
    users: int  # Target number of users
    spawn_rate: float  # Users per second to spawn/stop
    name: str = ""  # Optional stage name


class BaseLoadProfile(LoadTestShape, ABC):
    """Base class for load test profiles."""

    def __init__(self):
        """Initialize the load profile with no start time."""
        super().__init__()
        self.start_time = None

    def tick(self) -> tuple[int, float] | None:
        """Determine current users and spawn rate.

        Returns:
            Tuple of (user_count, spawn_rate) or None to stop
        """
        if self.start_time is None:
            self.start_time = time.time()

        run_time = time.time() - self.start_time
        return self.get_target_users(run_time)

    @abstractmethod
    def get_target_users(self, run_time: float) -> tuple[int, float] | None:
        """Get target users for current run time."""


class SteadyLoadProfile(BaseLoadProfile):
    """Steady load profile - constant number of users."""

    def __init__(self, users: int = 100, duration: int = 300, spawn_rate: float = 10):
        """Initialize steady load profile with constant users."""
        super().__init__()
        self.users = users
        self.duration = duration
        self.spawn_rate = spawn_rate

    def get_target_users(self, run_time: float) -> tuple[int, float] | None:
        """Maintain steady number of users."""
        if run_time < self.duration:
            return (self.users, self.spawn_rate)
        return None


class RampUpLoadProfile(BaseLoadProfile):
    """Ramp-up load profile - gradually increase users."""

    def __init__(
        self,
        start_users: int = 1,
        end_users: int = 100,
        ramp_time: int = 300,
        hold_time: int = 300,
        spawn_rate: float = 1,
    ):
        """Initialize ramp-up profile with gradual user increase."""
        super().__init__()
        self.start_users = start_users
        self.end_users = end_users
        self.ramp_time = ramp_time
        self.hold_time = hold_time
        self.spawn_rate = spawn_rate

    def get_target_users(self, run_time: float) -> tuple[int, float] | None:
        """Gradually increase users then hold steady."""
        if run_time < self.ramp_time:
            # Ramp up phase
            progress = run_time / self.ramp_time
            current_users = int(
                self.start_users + (self.end_users - self.start_users) * progress
            )
            return (current_users, self.spawn_rate)
        if run_time < self.ramp_time + self.hold_time:
            # Hold phase
            return (self.end_users, self.spawn_rate)
        return None


class SpikeLoadProfile(BaseLoadProfile):
    """Spike load profile - sudden increase in users."""

    def __init__(
        self,
        baseline_users: int = 50,
        spike_users: int = 500,
        baseline_time: int = 120,
        spike_time: int = 60,
        recovery_time: int = 120,
        spawn_rate: float = 50,
    ):
        """Initialize spike profile with sudden user increase."""
        super().__init__()
        self.baseline_users = baseline_users
        self.spike_users = spike_users
        self.baseline_time = baseline_time
        self.spike_time = spike_time
        self.recovery_time = recovery_time
        self.spawn_rate = spawn_rate

    def get_target_users(self, run_time: float) -> tuple[int, float] | None:
        """Create spike pattern."""
        if run_time < self.baseline_time:
            # Initial baseline
            return (self.baseline_users, self.spawn_rate / 10)
        if run_time < self.baseline_time + self.spike_time:
            # Spike phase
            return (self.spike_users, self.spawn_rate)
        if run_time < self.baseline_time + self.spike_time + self.recovery_time:
            # Recovery phase
            return (self.baseline_users, self.spawn_rate)
        return None


class StepLoadProfile(BaseLoadProfile):
    """Step load profile - increase users in steps."""

    def __init__(self, stages: list[LoadStage]):
        """Initialize step profile with stage-based user levels."""
        super().__init__()
        self.stages = stages

    def get_target_users(self, run_time: float) -> tuple[int, float] | None:
        """Step through user levels."""
        current_time = 0

        for stage in self.stages:
            if run_time <= current_time + stage.duration:
                return (stage.users, stage.spawn_rate)
            current_time += stage.duration

        return None


class WaveLoadProfile(BaseLoadProfile):
    """Wave load profile - sinusoidal user pattern."""

    def __init__(
        self,
        min_users: int = 10,
        max_users: int = 100,
        wave_duration: int = 300,
        _total_duration: int = 1800,
        spawn_rate: float = 5,
    ):
        """Initialize wave profile with sinusoidal user pattern."""
        super().__init__()
        self.min_users = min_users
        self.max_users = max_users
        self.wave_duration = wave_duration
        self._total_duration = _total_duration
        self.spawn_rate = spawn_rate

    def get_target_users(self, run_time: float) -> tuple[int, float] | None:
        """Create wave pattern."""
        if run_time < self._total_duration:
            # Sinusoidal wave
            amplitude = (self.max_users - self.min_users) / 2
            midpoint = self.min_users + amplitude
            wave_position = (run_time / self.wave_duration) * 2 * math.pi
            current_users = int(midpoint + amplitude * math.sin(wave_position))
            return (current_users, self.spawn_rate)
        return None


class DoubleSpike(BaseLoadProfile):
    """Double spike profile - two spikes with recovery."""

    def __init__(self):
        """Initialize double spike profile with two separate spikes."""
        super().__init__()
        self.stages = [
            LoadStage(duration=120, users=50, spawn_rate=5, name="warmup"),
            LoadStage(duration=60, users=300, spawn_rate=50, name="first_spike"),
            LoadStage(duration=120, users=50, spawn_rate=20, name="recovery"),
            LoadStage(duration=60, users=400, spawn_rate=50, name="second_spike"),
            LoadStage(duration=120, users=50, spawn_rate=20, name="cooldown"),
        ]

    def get_target_users(self, run_time: float) -> tuple[int, float] | None:
        """Execute double spike pattern."""
        current_time = 0

        for stage in self.stages:
            if run_time <= current_time + stage.duration:
                return (stage.users, stage.spawn_rate)
            current_time += stage.duration

        return None


class BreakpointLoadProfile(BaseLoadProfile):
    """Breakpoint load profile - find system breaking point."""

    def __init__(
        self,
        start_users: int = 10,
        user_increment: int = 10,
        step_duration: int = 60,
        max_users: int = 1000,
        spawn_rate: float = 5,
    ):
        """Initialize breakpoint profile to find system limits."""
        super().__init__()
        self.start_users = start_users
        self.user_increment = user_increment
        self.step_duration = step_duration
        self.max_users = max_users
        self.spawn_rate = spawn_rate

    def get_target_users(self, run_time: float) -> tuple[int, float] | None:
        """Incrementally increase load until breakpoint."""
        current_step = int(run_time / self.step_duration)
        current_users = self.start_users + (current_step * self.user_increment)

        if current_users <= self.max_users:
            return (current_users, self.spawn_rate)
        return None


# Predefined load profiles for common scenarios
LOAD_PROFILES = {
    "steady": SteadyLoadProfile(users=100, duration=300),
    "ramp_up": RampUpLoadProfile(
        start_users=1, end_users=200, ramp_time=300, hold_time=300
    ),
    "spike": SpikeLoadProfile(baseline_users=50, spike_users=500),
    "step": StepLoadProfile(
        [
            LoadStage(duration=120, users=50, spawn_rate=5),
            LoadStage(duration=120, users=100, spawn_rate=10),
            LoadStage(duration=120, users=200, spawn_rate=15),
            LoadStage(duration=120, users=400, spawn_rate=20),
        ]
    ),
    "wave": WaveLoadProfile(min_users=20, max_users=200, wave_duration=300),
    "double_spike": DoubleSpike(),
    "breakpoint": BreakpointLoadProfile(start_users=10, user_increment=10),
}


def get_load_profile(profile_name: str) -> BaseLoadProfile | None:
    """Get a predefined load profile by name.

    Args:
        profile_name: Name of the load profile

    Returns:
        Load profile instance or None if not found
    """
    return LOAD_PROFILES.get(profile_name)


def create_custom_step_profile(steps: list[dict[str, any]]) -> StepLoadProfile:
    """Create a custom step load profile.

    Args:
        steps: List of step configurations with duration, users, and spawn_rate

    Returns:
        Custom step load profile
    """
    stages = [
        LoadStage(
            duration=step["duration"],
            users=step["users"],
            spawn_rate=step.get("spawn_rate", 10),
            name=step.get("name", ""),
        )
        for step in steps
    ]
    return StepLoadProfile(stages)
