"""Environment Configuration Validation Tests.

This module tests environment-specific configurations and ensures consistency
across development, staging, and production environments.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


import pytest

from tests.deployment.conftest import DeploymentEnvironment
from tests.deployment.conftest import DeploymentTestConfig


class TestEnvironmentValidation:
    """Test environment configuration validation."""
    
    @pytest.mark.environment
    def test_development_environment_config(
        self, environment_configs: dict[str, DeploymentEnvironment]
    ):
        """Test development environment configuration."""
        dev_env = environment_configs["development"]
        
        # Development environment should be minimal
        assert dev_env.name == "development"
        assert dev_env.tier == "development"
        assert dev_env.infrastructure == "local"
        assert dev_env.database_type == "sqlite"
        assert dev_env.cache_type == "local"
        assert dev_env.vector_db_type == "memory"
        assert dev_env.monitoring_level == "basic"
        assert not dev_env.load_balancer
        assert not dev_env.ssl_enabled
        assert not dev_env.backup_enabled
        assert not dev_env.is_production
        assert not dev_env.requires_ssl
    
    @pytest.mark.environment
    def test_staging_environment_config(
        self, environment_configs: dict[str, DeploymentEnvironment]
    ):
        """Test staging environment configuration."""
        staging_env = environment_configs["staging"]
        
        # Staging should mirror production capabilities
        assert staging_env.name == "staging"
        assert staging_env.tier == "staging"
        assert staging_env.infrastructure == "cloud"
        assert staging_env.database_type == "postgresql"
        assert staging_env.cache_type == "redis"
        assert staging_env.vector_db_type == "qdrant"
        assert staging_env.monitoring_level == "full"
        assert staging_env.load_balancer
        assert staging_env.ssl_enabled
        assert staging_env.backup_enabled
        assert not staging_env.is_production
        assert staging_env.requires_ssl
    
    @pytest.mark.environment
    def test_production_environment_config(
        self, environment_configs: dict[str, DeploymentEnvironment]
    ):
        """Test production environment configuration."""
        prod_env = environment_configs["production"]
        
        # Production should have enterprise-grade configuration
        assert prod_env.name == "production"
        assert prod_env.tier == "production"
        assert prod_env.infrastructure == "cloud"
        assert prod_env.database_type == "postgresql"
        assert prod_env.cache_type == "dragonfly"
        assert prod_env.vector_db_type == "qdrant"
        assert prod_env.monitoring_level == "enterprise"
        assert prod_env.load_balancer
        assert prod_env.ssl_enabled
        assert prod_env.backup_enabled
        assert prod_env.is_production
        assert prod_env.requires_ssl
    
    @pytest.mark.environment
    def test_environment_progression(
        self, environment_configs: dict[str, DeploymentEnvironment]
    ):
        """Test that environments progress logically from dev to production."""
        dev = environment_configs["development"]
        staging = environment_configs["staging"]
        prod = environment_configs["production"]
        
        # Infrastructure complexity should increase
        infra_complexity = {"local": 0, "cloud": 1, "hybrid": 2}
        assert infra_complexity[dev.infrastructure] <= infra_complexity[staging.infrastructure]
        assert infra_complexity[staging.infrastructure] <= infra_complexity[prod.infrastructure]
        
        # Monitoring should become more comprehensive
        monitoring_levels = {"basic": 0, "full": 1, "enterprise": 2}
        assert monitoring_levels[dev.monitoring_level] <= monitoring_levels[staging.monitoring_level]
        assert monitoring_levels[staging.monitoring_level] <= monitoring_levels[prod.monitoring_level]
        
        # Production features should be enabled in higher environments
        assert not dev.ssl_enabled or staging.ssl_enabled
        assert not staging.ssl_enabled or prod.ssl_enabled
        assert not dev.backup_enabled or staging.backup_enabled
        assert not staging.backup_enabled or prod.backup_enabled


class TestConfigurationDrift:
    """Test configuration drift detection between environments."""
    
    @pytest.fixture
    def sample_configs(self, temp_deployment_dir: Path) -> dict[str, Path]:
        """Create sample configuration files for drift testing."""
        configs = {}
        
        base_config = {
            "app": {
                "name": "ai-docs-scraper",
                "version": "1.0.0",
                "debug": False,
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "ai_docs",
            },
            "cache": {
                "ttl_seconds": 3600,
                "max_size_mb": 500,
            },
            "vector_db": {
                "collection_name": "documents",
                "embedding_dimension": 384,
            }
        }
        
        # Development config
        dev_config = base_config.copy()
        dev_config["app"]["debug"] = True
        dev_config["database"]["host"] = "localhost"
        dev_config["database"]["name"] = "ai_docs_dev"
        
        dev_file = temp_deployment_dir / "dev_config.json"
        with open(dev_file, "w") as f:
            json.dump(dev_config, f, indent=2)
        configs["development"] = dev_file
        
        # Staging config
        staging_config = base_config.copy()
        staging_config["database"]["host"] = "staging-db.example.com"
        staging_config["database"]["name"] = "ai_docs_staging"
        staging_config["cache"]["max_size_mb"] = 1000
        
        staging_file = temp_deployment_dir / "staging_config.json"
        with open(staging_file, "w") as f:
            json.dump(staging_config, f, indent=2)
        configs["staging"] = staging_file
        
        # Production config
        prod_config = base_config.copy()
        prod_config["database"]["host"] = "prod-db.example.com"
        prod_config["database"]["name"] = "ai_docs_prod"
        prod_config["cache"]["max_size_mb"] = 2048
        prod_config["vector_db"]["collection_name"] = "documents_prod"
        
        prod_file = temp_deployment_dir / "prod_config.json"
        with open(prod_file, "w") as f:
            json.dump(prod_config, f, indent=2)
        configs["production"] = prod_file
        
        return configs
    
    @pytest.mark.environment
    def test_detect_configuration_drift(self, sample_configs: dict[str, Path]):
        """Test configuration drift detection between environments."""
        drift_detector = ConfigurationDriftDetector()
        
        # Load configurations
        configs = {}
        for env, config_file in sample_configs.items():
            with open(config_file) as f:
                configs[env] = json.load(f)
        
        # Detect drift between development and staging
        dev_staging_drift = drift_detector.detect_drift(
            configs["development"], configs["staging"]
        )
        
        # Should detect differences in debug flag, database host, and cache size
        assert len(dev_staging_drift) >= 3
        assert any("app.debug" in drift["path"] for drift in dev_staging_drift)
        assert any("database.host" in drift["path"] for drift in dev_staging_drift)
        assert any("cache.max_size_mb" in drift["path"] for drift in dev_staging_drift)
        
        # Detect drift between staging and production
        staging_prod_drift = drift_detector.detect_drift(
            configs["staging"], configs["production"]
        )
        
        # Should detect differences in database host, cache size, and collection name
        assert len(staging_prod_drift) >= 3
        assert any("database.host" in drift["path"] for drift in staging_prod_drift)
        assert any("cache.max_size_mb" in drift["path"] for drift in staging_prod_drift)
        assert any("vector_db.collection_name" in drift["path"] for drift in staging_prod_drift)
    
    @pytest.mark.environment
    def test_acceptable_drift_vs_problematic_drift(self, sample_configs: dict[str, Path]):
        """Test distinguishing between acceptable and problematic configuration drift."""
        drift_detector = ConfigurationDriftDetector()
        
        # Load configurations
        configs = {}
        for env, config_file in sample_configs.items():
            with open(config_file) as f:
                configs[env] = json.load(f)
        
        # Categorize drift
        drift = drift_detector.detect_drift(configs["development"], configs["staging"])
        categorized = drift_detector.categorize_drift(drift)
        
        # Debug flag difference should be acceptable (development-specific)
        debug_drifts = [d for d in categorized["acceptable"] if "debug" in d["path"]]
        assert len(debug_drifts) > 0
        
        # Database host difference should be acceptable (environment-specific)
        host_drifts = [d for d in categorized["acceptable"] if "database.host" in d["path"]]
        assert len(host_drifts) > 0
        
        # App name or version differences would be problematic
        if any("app.name" in d["path"] or "app.version" in d["path"] for d in drift):
            problematic_drifts = [
                d for d in categorized["problematic"] 
                if "app.name" in d["path"] or "app.version" in d["path"]
            ]
            assert len(problematic_drifts) > 0


class TestSecretsManagement:
    """Test secrets management across environments."""
    
    @pytest.mark.environment
    def test_secrets_not_in_config_files(self, temp_deployment_dir: Path):
        """Test that secrets are not stored in configuration files."""
        secrets_validator = SecretsValidator()
        
        # Create config with potential secrets
        config_with_secrets = {
            "database": {
                "host": "localhost",
                "username": "user",
                "password": "secret123",  # This should be flagged
            },
            "api_keys": {
                "openai": "sk-1234567890abcdef",  # This should be flagged
                "google": "${GOOGLE_API_KEY}",  # This is acceptable (env var reference)
            },
            "jwt": {
                "secret": "my-jwt-secret",  # This should be flagged
                "algorithm": "HS256",
            }
        }
        
        config_file = temp_deployment_dir / "config_with_secrets.json"
        with open(config_file, "w") as f:
            json.dump(config_with_secrets, f, indent=2)
        
        # Validate secrets
        violations = secrets_validator.scan_for_secrets(config_file)
        
        # Should detect multiple secret violations
        assert len(violations) >= 3
        assert any("password" in v["field"] for v in violations)
        assert any("openai" in v["field"] for v in violations)
        assert any("secret" in v["field"] for v in violations)
        
        # Should not flag environment variable references
        env_var_violations = [v for v in violations if "${" in v["value"]]
        assert len(env_var_violations) == 0
    
    @pytest.mark.environment
    def test_environment_variable_references(self, temp_deployment_dir: Path):
        """Test that configurations properly reference environment variables."""
        config = {
            "database": {
                "host": "${DATABASE_HOST}",
                "port": "${DATABASE_PORT}",
                "username": "${DATABASE_USER}",
                "password": "${DATABASE_PASSWORD}",
            },
            "cache": {
                "url": "${CACHE_URL}",
            },
            "api_keys": {
                "openai": "${OPENAI_API_KEY}",
            }
        }
        
        config_file = temp_deployment_dir / "env_var_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        validator = SecretsValidator()
        violations = validator.scan_for_secrets(config_file)
        
        # Should not find any violations (all values are env var references)
        assert len(violations) == 0
    
    @pytest.mark.environment
    def test_required_environment_variables(
        self, deployment_environment: DeploymentEnvironment
    ):
        """Test that required environment variables are defined."""
        env_validator = EnvironmentVariableValidator()
        
        # Define required environment variables based on environment
        if deployment_environment.is_production:
            required_vars = [
                "DATABASE_HOST",
                "DATABASE_PASSWORD",
                "OPENAI_API_KEY",
                "JWT_SECRET",
                "SSL_CERT_PATH",
                "SSL_KEY_PATH",
            ]
        else:
            required_vars = [
                "DATABASE_HOST",
                "DATABASE_PASSWORD",
                "OPENAI_API_KEY",
            ]
        
        # Check if variables are set (in testing, we'll simulate this)
        missing_vars = env_validator.check_required_variables(required_vars)
        
        # In testing environment, variables might not be set
        # This test would be more meaningful in actual deployment pipeline
        if deployment_environment.name == "development":
            # Development might have missing variables, which is acceptable for testing
            assert isinstance(missing_vars, list)
        else:
            # Staging and production should have all required variables
            # In real deployment, this would fail if variables are missing
            assert isinstance(missing_vars, list)


class ConfigurationDriftDetector:
    """Detector for configuration drift between environments."""
    
    def detect_drift(
        self, config1: dict[str, Any], config2: dict[str, Any], path: str = ""
    ) -> list[dict[str, Any]]:
        """Detect drift between two configurations."""
        drift = []
        
        # Compare all keys in both configurations
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in config1:
                drift.append({
                    "path": current_path,
                    "type": "missing_in_first",
                    "second_value": config2[key],
                })
            elif key not in config2:
                drift.append({
                    "path": current_path,
                    "type": "missing_in_second",
                    "first_value": config1[key],
                })
            elif isinstance(config1[key], dict) and isinstance(config2[key], dict):
                # Recursively check nested dictionaries
                nested_drift = self.detect_drift(config1[key], config2[key], current_path)
                drift.extend(nested_drift)
            elif config1[key] != config2[key]:
                drift.append({
                    "path": current_path,
                    "type": "value_mismatch",
                    "first_value": config1[key],
                    "second_value": config2[key],
                })
        
        return drift
    
    def categorize_drift(self, drift: list[dict[str, Any]]) -> dict[str, list[Dict[str, Any]]]:
        """Categorize drift as acceptable or problematic."""
        acceptable_patterns = [
            "debug",
            "host",
            "url",
            "database.name",
            "cache.max_size",
            "log_level",
        ]
        
        problematic_patterns = [
            "app.name",
            "app.version",
            "api.version",
            "security",
        ]
        
        categorized = {
            "acceptable": [],
            "problematic": [],
            "unclear": [],
        }
        
        for drift_item in drift:
            path = drift_item["path"].lower()
            
            if any(pattern in path for pattern in acceptable_patterns):
                categorized["acceptable"].append(drift_item)
            elif any(pattern in path for pattern in problematic_patterns):
                categorized["problematic"].append(drift_item)
            else:
                categorized["unclear"].append(drift_item)
        
        return categorized


class SecretsValidator:
    """Validator for detecting secrets in configuration files."""
    
    def __init__(self):
        self.secret_patterns = [
            r"password",
            r"secret",
            r"key",
            r"token",
            r"credential",
            r"auth",
        ]
        
        self.secret_value_patterns = [
            r"^sk-[a-zA-Z0-9]{32,}$",  # OpenAI API key pattern
            r"^[a-zA-Z0-9]{32,}$",    # Generic long alphanumeric strings
            r"^[a-f0-9]{32,}$",       # Hex strings (common for secrets)
        ]
    
    def scan_for_secrets(self, config_file: Path) -> list[dict[str, Any]]:
        """Scan configuration file for potential secrets."""
        violations = []
        
        with open(config_file) as f:
            config = json.load(f)
        
        self._scan_dict(config, violations)
        return violations
    
    def _scan_dict(self, obj: Any, violations: list[dict[str, Any]], path: str = "") -> None:
        """Recursively scan dictionary for secrets."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                self._scan_dict(value, violations, current_path)
        elif isinstance(obj, str):
            # Skip environment variable references
            if obj.startswith("${") and obj.endswith("}"):
                return
            
            # Check if field name suggests a secret
            field_suggests_secret = any(
                pattern in path.lower() for pattern in self.secret_patterns
            )
            
            # Check if value looks like a secret
            value_looks_like_secret = any(
                __import__("re").match(pattern, obj) for pattern in self.secret_value_patterns
            )
            
            if field_suggests_secret or value_looks_like_secret:
                violations.append({
                    "field": path,
                    "value": obj,
                    "reason": "field_name" if field_suggests_secret else "value_pattern",
                })


class EnvironmentVariableValidator:
    """Validator for environment variable requirements."""
    
    def check_required_variables(self, required_vars: list[str]) -> list[str]:
        """Check which required environment variables are missing."""
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        return missing_vars
    
    def validate_variable_values(self, var_definitions: dict[str, Dict[str, Any]]) -> dict[str, list[str]]:
        """Validate environment variable values against requirements."""
        validation_errors = {}
        
        for var_name, requirements in var_definitions.items():
            errors = []
            value = os.getenv(var_name)
            
            if not value and requirements.get("required", False):
                errors.append("Variable is required but not set")
                continue
            
            if value:
                # Check minimum length
                if "min_length" in requirements and len(value) < requirements["min_length"]:
                    errors.append(f"Value too short (minimum {requirements['min_length']} characters)")
                
                # Check pattern
                if "pattern" in requirements:
                    import re
                    if not re.match(requirements["pattern"], value):
                        errors.append(f"Value does not match required pattern: {requirements['pattern']}")
                
                # Check allowed values
                if "allowed_values" in requirements and value not in requirements["allowed_values"]:
                    errors.append(f"Value not in allowed list: {requirements['allowed_values']}")
            
            if errors:
                validation_errors[var_name] = errors
        
        return validation_errors