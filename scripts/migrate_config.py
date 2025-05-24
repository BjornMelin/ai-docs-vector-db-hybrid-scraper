#!/usr/bin/env python3
"""Script to migrate existing configuration to the unified config system."""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ConfigLoader
from src.config import UnifiedConfig


def migrate_documentation_sites():
    """Migrate documentation sites from old JSON format."""
    old_path = Path("config/documentation-sites.json")
    if not old_path.exists():
        print(f"No documentation sites file found at {old_path}")
        return []

    print(f"Migrating documentation sites from {old_path}")
    sites = ConfigLoader.load_documentation_sites(old_path)
    print(f"  - Found {len(sites)} sites")
    return sites


def migrate_env_variables():
    """Collect environment variables that match our pattern."""
    env_vars = {}

    # Old pattern variables
    old_mappings = {
        "OPENAI_API_KEY": "openai.api_key",
        "FIRECRAWL_API_KEY": "firecrawl.api_key",
        "QDRANT_URL": "qdrant.url",
        "QDRANT_API_KEY": "qdrant.api_key",
        "REDIS_URL": "cache.redis_url",
    }

    for old_key, _new_path in old_mappings.items():
        if old_key in os.environ:
            env_vars[old_key] = os.environ[old_key]
            print(f"  - Found {old_key}")

    # New pattern variables
    prefix = "AI_DOCS__"
    for key, value in os.environ.items():
        if key.startswith(prefix):
            env_vars[key] = value
            print(f"  - Found {key}")

    return env_vars


def main():  # noqa: PLR0912
    """Main migration function."""
    print("AI Documentation Vector DB Configuration Migration\n")

    # Create new unified config
    config = UnifiedConfig()

    # Migrate documentation sites
    print("1. Migrating documentation sites...")
    sites = migrate_documentation_sites()
    if sites:
        config.documentation_sites = sites

    # Check for environment variables
    print("\n2. Checking environment variables...")
    env_vars = migrate_env_variables()

    # Save configurations
    print("\n3. Creating configuration files...")

    # Save main config
    config_path = Path("config.json")
    config.save_to_file(config_path, format="json")
    print(f"  ✓ Created {config_path}")

    # Create .env.example
    env_example_path = Path(".env.example")
    ConfigLoader.create_env_template(env_example_path)
    print(f"  ✓ Created {env_example_path}")

    # Create .env if we found variables
    if env_vars:
        env_path = Path(".env")
        if not env_path.exists():
            with open(env_path, "w") as f:
                f.write("# AI Documentation Vector DB Configuration\n")
                f.write("# Auto-generated from existing environment\n\n")

                # Write old-style variables
                for key, value in env_vars.items():
                    if not key.startswith("AI_DOCS__"):
                        # Convert to new format
                        if key == "OPENAI_API_KEY":
                            f.write(f"AI_DOCS__OPENAI__API_KEY={value}\n")
                        elif key == "FIRECRAWL_API_KEY":
                            f.write(f"AI_DOCS__FIRECRAWL__API_KEY={value}\n")
                        elif key == "QDRANT_URL":
                            f.write(f"AI_DOCS__QDRANT__URL={value}\n")
                        elif key == "QDRANT_API_KEY":
                            f.write(f"AI_DOCS__QDRANT__API_KEY={value}\n")
                        elif key == "REDIS_URL":
                            f.write(f"AI_DOCS__CACHE__REDIS_URL={value}\n")
                    else:
                        # Keep new format
                        f.write(f"{key}={value}\n")

            print(f"  ✓ Created {env_path} with migrated variables")
        else:
            print(f"  ! {env_path} already exists, skipping")

    # Validate the configuration
    print("\n4. Validating configuration...")
    is_valid, issues = ConfigLoader.validate_config(config)

    if is_valid:
        print("  ✓ Configuration is valid!")
    else:
        print("  ! Configuration has issues:")
        for issue in issues:
            print(f"    - {issue}")

    print("\n✅ Migration complete!")
    print("\nNext steps:")
    print("1. Review the generated config.json")
    print("2. Copy .env.example to .env and fill in your API keys")
    print("3. Run 'python -m src.manage_config validate' to check your configuration")


if __name__ == "__main__":
    main()
