#!/usr/bin/env python3
"""Simple configuration validator using Pydantic."""

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class CacheConfig(BaseModel):
    """Cache configuration."""

    ttl: int = Field(ge=0, le=86400)
    max_size: int = Field(gt=0)


class PerformanceConfig(BaseModel):
    """Performance configuration."""

    batch_size: int = Field(gt=0, le=1000)
    max_workers: int = Field(gt=0, le=32)
    timeout: int = Field(gt=0)


class Config(BaseModel):
    """Main configuration schema."""

    environment: str = Field(..., pattern="^(development|staging|production)$")
    debug: bool = False
    cache: CacheConfig
    performance: PerformanceConfig


def validate_config(config_path: Path) -> bool:
    """Validate a configuration file."""
    try:
        with config_path.open() as f:
            data = json.load(f)

        Config(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"❌ Invalid: {config_path} - {e}")
        return False
    else:
        print(f"✅ Valid: {config_path}")
        return True


def main():
    """Validate all configuration files."""
    config_dir = Path("config")
    if not config_dir.exists():
        print("❌ Config directory not found")
        return 1

    valid = all(
        validate_config(f)
        for f in config_dir.rglob("*.json")
        if "template" in f.parent.name
    )

    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
