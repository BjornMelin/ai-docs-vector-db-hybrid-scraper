"""Application profile definitions for FastAPI app composition."""

from __future__ import annotations

from enum import Enum

from src.architecture.modes import ApplicationMode, resolve_mode


class AppProfile(str, Enum):
    """Supported application profiles for API composition."""

    SIMPLE = "simple"
    ENTERPRISE = "enterprise"

    def to_mode(self) -> ApplicationMode:
        """Convert the profile to the corresponding architecture mode."""

        return ApplicationMode(self.value)

    @classmethod
    def from_mode(cls, mode: ApplicationMode) -> AppProfile:
        """Create a profile from an :class:`ApplicationMode`."""

        return cls(mode.value)


def detect_profile() -> AppProfile:
    """Detect the current application profile using architecture mode settings."""

    return AppProfile.from_mode(resolve_mode())


__all__ = ["AppProfile", "detect_profile"]
