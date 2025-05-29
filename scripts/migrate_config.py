#!/usr/bin/env python3
"""Enhanced script to migrate existing configuration to the unified config system.

This script now handles migration of:
- Legacy environment variables (OPENAI_API_KEY, FIRECRAWL_API_KEY, etc.)
- AI_DOCS__ prefixed configuration variables (unified config system)
- FastMCP streaming configuration (FASTMCP_TRANSPORT, etc.)
- Docker Compose environment variables (QDRANT__, DRAGONFLY_)
- Testing environment variables (RUN_REAL_INTEGRATION_TESTS, etc.)
- Rate limiting configuration (RATE_LIMIT_*)
- Browser automation settings
- HyDE, reranking, and Query API configuration
- Collection aliases and deployment strategy settings

Updated to reflect V1 architecture with DragonflyDB caching, BGE reranking,
Crawl4AI integration, and FastMCP 2.0 streaming support.
"""

import os
from pathlib import Path

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

    # FastMCP variables
    fastmcp_keys = [
        "FASTMCP_TRANSPORT",
        "FASTMCP_HOST",
        "FASTMCP_PORT",
        "FASTMCP_BUFFER_SIZE",
        "FASTMCP_MAX_RESPONSE_SIZE",
    ]

    for key in fastmcp_keys:
        if key in os.environ:
            env_vars[key] = os.environ[key]
            print(f"  - Found {key}")

    # Docker compose environment variables
    docker_keys = [
        "QDRANT__SERVICE__HTTP_PORT",
        "QDRANT__SERVICE__GRPC_PORT",
        "QDRANT__LOG_LEVEL",
        "QDRANT__STORAGE__ON_DISK_PAYLOAD",
        "QDRANT__STORAGE__QUANTIZATION__ALWAYS_RAM",
        "QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS",
        "DRAGONFLY_THREADS",
        "DRAGONFLY_MEMORY_LIMIT",
        "DRAGONFLY_SNAPSHOT_INTERVAL",
    ]

    for key in docker_keys:
        if key in os.environ:
            env_vars[key] = os.environ[key]
            print(f"  - Found {key}")

    # Testing environment variables
    test_keys = ["RUN_REAL_INTEGRATION_TESTS", "CRAWL4AI_TIMEOUT", "REQUEST_TIMEOUT"]

    for key in test_keys:
        if key in os.environ:
            env_vars[key] = os.environ[key]
            print(f"  - Found {key}")

    # Rate limiting variables
    rate_limit_keys = [
        "RATE_LIMIT_OPENAI_MAX_CALLS",
        "RATE_LIMIT_OPENAI_TIME_WINDOW",
        "RATE_LIMIT_FIRECRAWL_MAX_CALLS",
        "RATE_LIMIT_FIRECRAWL_TIME_WINDOW",
    ]

    for key in rate_limit_keys:
        if key in os.environ:
            env_vars[key] = os.environ[key]
            print(f"  - Found {key}")

    return env_vars


def main():  # noqa: PLR0912, PLR0915
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

                # Write environment variables in sections
                legacy_keys = []
                ai_docs_keys = []
                fastmcp_keys = []
                docker_keys = []
                test_keys = []
                rate_limit_keys = []

                # Categorize variables
                for key, value in env_vars.items():
                    if key.startswith("AI_DOCS__"):
                        ai_docs_keys.append((key, value))
                    elif key.startswith("FASTMCP_"):
                        fastmcp_keys.append((key, value))
                    elif key.startswith(("QDRANT__", "DRAGONFLY_")):
                        docker_keys.append((key, value))
                    elif key.startswith("RATE_LIMIT_"):
                        rate_limit_keys.append((key, value))
                    elif key in [
                        "RUN_REAL_INTEGRATION_TESTS",
                        "CRAWL4AI_TIMEOUT",
                        "REQUEST_TIMEOUT",
                    ]:
                        test_keys.append((key, value))
                    else:
                        legacy_keys.append((key, value))

                # Write legacy variables with conversion
                if legacy_keys:
                    f.write(
                        "# Legacy environment variables (migrated to AI_DOCS__ format)\n"
                    )
                    for key, value in legacy_keys:
                        if key == "OPENAI_API_KEY":
                            f.write(f"OPENAI_API_KEY={value}\n")
                            f.write(f"AI_DOCS__OPENAI__API_KEY={value}\n")
                        elif key == "FIRECRAWL_API_KEY":
                            f.write(f"FIRECRAWL_API_KEY={value}\n")
                            f.write(f"AI_DOCS__FIRECRAWL__API_KEY={value}\n")
                        elif key == "QDRANT_URL":
                            f.write(f"QDRANT_URL={value}\n")
                            f.write(f"AI_DOCS__QDRANT__URL={value}\n")
                        elif key == "QDRANT_API_KEY":
                            f.write(f"QDRANT_API_KEY={value}\n")
                            f.write(f"AI_DOCS__QDRANT__API_KEY={value}\n")
                        elif key == "REDIS_URL":
                            f.write(f"REDIS_URL={value}\n")
                            f.write(f"AI_DOCS__CACHE__REDIS_URL={value}\n")
                        else:
                            f.write(f"{key}={value}\n")
                    f.write("\n")

                # Write FastMCP variables
                if fastmcp_keys:
                    f.write("# FastMCP Configuration\n")
                    for key, value in fastmcp_keys:
                        f.write(f"{key}={value}\n")
                    f.write("\n")

                # Write AI_DOCS__ prefixed variables
                if ai_docs_keys:
                    f.write("# AI Documentation Scraper Configuration\n")
                    for key, value in ai_docs_keys:
                        f.write(f"{key}={value}\n")
                    f.write("\n")

                # Write Docker variables
                if docker_keys:
                    f.write("# Docker Compose Configuration\n")
                    for key, value in docker_keys:
                        f.write(f"{key}={value}\n")
                    f.write("\n")

                # Write test variables
                if test_keys:
                    f.write("# Testing Configuration\n")
                    for key, value in test_keys:
                        f.write(f"{key}={value}\n")
                    f.write("\n")

                # Write rate limit variables
                if rate_limit_keys:
                    f.write("# Rate Limiting Configuration\n")
                    for key, value in rate_limit_keys:
                        f.write(f"{key}={value}\n")
                    f.write("\n")

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

    print("\n✅ Enhanced migration complete!")
    print("\nNext steps:")
    print("1. Review the generated config.json")
    print("2. Copy .env.example to .env and fill in your API keys")
    print("3. Configure advanced features:")
    print("   - FastMCP streaming (FASTMCP_TRANSPORT=streamable-http)")
    print("   - HyDE enhancement (AI_DOCS__HYDE__ENABLED=true)")
    print("   - BGE reranking (AI_DOCS__RERANKING__ENABLED=true)")
    print("   - DragonflyDB caching (AI_DOCS__CACHE__ENABLE_REDIS_CACHE=true)")
    print("   - Browser automation hierarchy settings")
    print("4. Run 'python -m src.manage_config validate' to check your configuration")
    print("5. Test MCP streaming: 'uv run python src/unified_mcp_server.py'")
    print("6. Monitor performance with enhanced metrics and caching")


if __name__ == "__main__":
    main()
