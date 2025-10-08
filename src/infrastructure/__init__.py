"""Infrastructure layer for external dependencies and framework concerns."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .client_manager import ClientManager

__all__ = ["ClientManager"]


def __getattr__(name: str):
    if name == "ClientManager":
        # pylint: disable=import-outside-toplevel
        from .client_manager import ClientManager as _ClientManager

        return _ClientManager
    raise AttributeError(name)
