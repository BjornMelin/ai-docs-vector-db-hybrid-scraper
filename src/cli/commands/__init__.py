"""CLI command modules for the advanced interface.

This package contains all command group implementations:
- config: Configuration management commands
- database: Vector database operations
- batch: Batch processing operations
- setup: Interactive configuration wizard
- task_runner: Developer automation commands ported from the legacy CLI
"""

from . import batch, config, database, setup, task_runner


__all__ = ["batch", "config", "database", "setup", "task_runner"]
