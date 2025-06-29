#!/usr/bin/env python3
"""Configuration validation script for AI Docs Vector DB."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import click


class ConfigValidator:
    """Validates project configuration and environment setup."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.project_root = Path(__file__).parent.parent
        
    def validate_environment_variables(self) -> bool:
        """Validate required environment variables."""
        click.echo("🔍 Validating environment variables...")
        
        # Check for .env files
        env_files = [".env", ".env.local", ".env.example"]
        found_env = False
        
        for env_file in env_files:
            if (self.project_root / env_file).exists():
                found_env = True
                click.echo(f"  ✅ Found {env_file}")
                
        if not found_env:
            self.errors.append("No environment files found (.env, .env.local)")
            
        # Validate key environment variables
        ai_docs_mode = os.getenv("AI_DOCS__MODE")
        if ai_docs_mode:
            if ai_docs_mode in ["simple", "enterprise"]:
                click.echo(f"  ✅ AI_DOCS__MODE: {ai_docs_mode}")
            else:
                self.errors.append(f"Invalid AI_DOCS__MODE: {ai_docs_mode}")
        else:
            self.warnings.append("AI_DOCS__MODE not set (defaults to 'simple')")
            
        return len(self.errors) == 0
        
    def validate_dependencies(self) -> bool:
        """Validate that required dependencies are installed."""
        click.echo("📦 Validating dependencies...")
        
        try:
            import fastapi
            click.echo(f"  ✅ FastAPI: {fastapi.__version__}")
        except ImportError:
            self.errors.append("FastAPI not installed")
            
        try:
            import qdrant_client
            click.echo(f"  ✅ Qdrant Client: {qdrant_client.__version__}")
        except ImportError:
            self.errors.append("Qdrant Client not installed")
            
        try:
            import pytest
            click.echo(f"  ✅ Pytest: {pytest.__version__}")
        except ImportError:
            self.errors.append("Pytest not installed")
            
        try:
            import ruff
            click.echo(f"  ✅ Ruff available")
        except ImportError:
            self.errors.append("Ruff not installed")
            
        return len(self.errors) == 0
        
    def validate_services(self) -> bool:
        """Validate that external services are accessible."""
        click.echo("🔗 Validating services...")
        
        # Test Qdrant connection
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        try:
            import httpx
            response = httpx.get(f"{qdrant_url}/health", timeout=5.0)
            if response.status_code == 200:
                click.echo(f"  ✅ Qdrant accessible at {qdrant_url}")
            else:
                self.warnings.append(f"Qdrant at {qdrant_url} returned status {response.status_code}")
        except Exception as e:
            self.warnings.append(f"Cannot connect to Qdrant at {qdrant_url}: {e}")
            
        # Test Redis connection
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        try:
            import redis
            r = redis.from_url(redis_url)
            r.ping()
            click.echo(f"  ✅ Redis accessible at {redis_url}")
        except Exception as e:
            self.warnings.append(f"Cannot connect to Redis at {redis_url}: {e}")
            
        return True  # Service availability is not critical for validation
        
    def validate_project_structure(self) -> bool:
        """Validate project structure and key files."""
        click.echo("📁 Validating project structure...")
        
        required_files = [
            "pyproject.toml",
            "src/__init__.py",
            "src/api/main.py",
            "src/cli/main.py",
            "tests/conftest.py",
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                click.echo(f"  ✅ {file_path}")
            else:
                self.errors.append(f"Missing required file: {file_path}")
                
        return len(self.errors) == 0
        
    def validate_test_configuration(self) -> bool:
        """Validate test configuration and markers."""
        click.echo("🧪 Validating test configuration...")
        
        pytest_files = ["pytest.ini", "pytest-fast.ini", "pytest-modern.ini"]
        found_pytest = False
        
        for pytest_file in pytest_files:
            if (self.project_root / pytest_file).exists():
                found_pytest = True
                click.echo(f"  ✅ Found {pytest_file}")
                
        if not found_pytest:
            self.errors.append("No pytest configuration found")
            
        return len(self.errors) == 0
        
    def run_validation(self) -> bool:
        """Run complete validation suite."""
        click.echo("🔍 Running configuration validation...\n")
        
        validations = [
            self.validate_environment_variables,
            self.validate_dependencies,
            self.validate_project_structure,
            self.validate_test_configuration,
            self.validate_services,
        ]
        
        all_passed = True
        for validation in validations:
            try:
                result = validation()
                all_passed = all_passed and result
            except Exception as e:
                self.errors.append(f"Validation error: {e}")
                all_passed = False
            click.echo()
            
        # Report results
        click.echo("📊 Validation Summary:")
        
        if self.errors:
            click.echo("❌ Errors:")
            for error in self.errors:
                click.echo(f"  • {error}")
                
        if self.warnings:
            click.echo("⚠️  Warnings:")
            for warning in self.warnings:
                click.echo(f"  • {warning}")
                
        if all_passed and not self.errors:
            click.echo("✅ All validations passed!")
            return True
        else:
            click.echo("❌ Validation failed!")
            return False


@click.command()
@click.option('--strict', is_flag=True, help="Treat warnings as errors")
def main(strict: bool):
    """Validate project configuration and environment."""
    validator = ConfigValidator()
    success = validator.run_validation()
    
    if strict and validator.warnings:
        click.echo("💥 Strict mode: treating warnings as errors")
        success = False
        
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()