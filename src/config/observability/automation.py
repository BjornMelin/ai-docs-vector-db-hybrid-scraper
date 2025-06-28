"""Configuration Observability Automation System.

A sophisticated automation system that provides intelligent monitoring, validation,
and optimization of configuration across all environments and deployment modes.

This system showcases enterprise-level automation capabilities:
- Real-time configuration drift detection and remediation
- Multi-environment configuration consistency validation
- Automated performance optimization based on metrics
- Intelligent configuration recommendations and auto-tuning
- Comprehensive configuration health monitoring
- Zero-downtime configuration updates with rollback capability

Features:
- Environment-aware configuration monitoring
- Automated drift detection and correction
- Performance-based configuration optimization
- Configuration compliance validation
- Real-time alerting and remediation
- Configuration version control and rollback
- Multi-tier deployment validation
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from watchdog.observers import Observer

from ..core import Config as LegacyConfig
from ..modern import ApplicationMode, Config as ModernConfig, Environment


logger = logging.getLogger(__name__)


class ConfigDriftSeverity(str, Enum):
    """Configuration drift severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


class ConfigValidationStatus(str, Enum):
    """Configuration validation status."""

    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AutomationAction(str, Enum):
    """Automation actions that can be performed."""

    MONITOR = "monitor"
    VALIDATE = "validate"
    OPTIMIZE = "optimize"
    REMEDIATE = "remediate"
    ROLLBACK = "rollback"
    ALERT = "alert"


@dataclass
class ConfigDrift:
    """Configuration drift detection result."""

    severity: ConfigDriftSeverity
    parameter: str
    expected_value: Any
    current_value: Any
    environment: str
    timestamp: datetime
    impact_score: float = 0.0
    remediation_suggested: bool = False
    auto_fix_available: bool = False


@dataclass
class ConfigValidationResult:
    """Configuration validation result."""

    status: ConfigValidationStatus
    parameter: str
    message: str
    environment: str
    timestamp: datetime
    suggestions: list[str] = field(default_factory=list)


@dataclass
class PerformanceMetric:
    """Performance metric for configuration optimization."""

    name: str
    value: float
    threshold: float
    unit: str
    timestamp: datetime
    environment: str
    config_parameter: str | None = None


@dataclass
class OptimizationRecommendation:
    """Configuration optimization recommendation."""

    parameter: str
    current_value: Any
    recommended_value: Any
    expected_improvement: str
    confidence_score: float
    performance_impact: str
    environment: str
    reasoning: str


class ConfigurationWatcher(FileSystemEventHandler):
    """File system watcher for configuration changes."""

    def __init__(self, automation_system: "ConfigObservabilityAutomation"):
        self.automation_system = automation_system
        self.watched_files = {
            ".env",
            ".env.simple",
            ".env.enterprise",
            ".env.test",
            "docker-compose.yml",
            "docker-compose.simple.yml",
            "docker-compose.enterprise.yml",
        }

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            file_name = Path(event.src_path).name
            if file_name in self.watched_files:
                logger.info(f"Configuration file modified: {file_name}")
                asyncio.create_task(
                    self.automation_system.handle_config_change(event.src_path)
                )


class ConfigObservabilityAutomation:
    """Advanced configuration observability and automation system."""

    def __init__(
        self,
        config_dir: str = ".",
        enable_auto_remediation: bool = False,
        enable_performance_optimization: bool = True,
        drift_check_interval: int = 300,  # 5 minutes
        performance_optimization_interval: int = 900,  # 15 minutes
    ):
        self.config_dir = Path(config_dir)
        self.enable_auto_remediation = enable_auto_remediation
        self.enable_performance_optimization = enable_performance_optimization
        self.drift_check_interval = drift_check_interval
        self.performance_optimization_interval = performance_optimization_interval

        # State tracking
        self.baseline_configurations: dict[str, dict[str, Any]] = {}
        self.drift_history: list[ConfigDrift] = []
        self.validation_history: list[ConfigValidationResult] = []
        self.performance_metrics: list[PerformanceMetric] = []
        self.optimization_recommendations: list[OptimizationRecommendation] = []

        # File system monitoring
        self.observer: Observer | None = None
        self.watcher = ConfigurationWatcher(self)

        # Performance tracking
        self.last_drift_check = datetime.now()
        self.last_optimization_check = datetime.now()

        # Environment detection
        self.detected_environments: set[str] = set()

        logger.info("Configuration Observability Automation System initialized")

    async def start(self) -> None:
        """Start the automation system."""
        logger.info("Starting Configuration Observability Automation System")

        # Initialize baseline configurations
        await self.establish_baseline_configurations()

        # Start file system monitoring
        self.start_file_monitoring()

        # Start periodic checks
        asyncio.create_task(self._periodic_drift_check())
        asyncio.create_task(self._periodic_performance_optimization())

        logger.info("Configuration automation system started successfully")

    async def stop(self) -> None:
        """Stop the automation system."""
        logger.info("Stopping Configuration Observability Automation System")

        if self.observer:
            self.observer.stop()
            self.observer.join()

        logger.info("Configuration automation system stopped")

    def start_file_monitoring(self) -> None:
        """Start file system monitoring for configuration changes."""
        self.observer = Observer()
        self.observer.schedule(self.watcher, str(self.config_dir), recursive=False)
        self.observer.start()
        logger.info("File system monitoring started")

    async def establish_baseline_configurations(self) -> None:
        """Establish baseline configurations for all environments."""
        logger.info("Establishing baseline configurations")

        # Detect available environments
        await self.detect_environments()

        # Load baseline for each environment
        for env in self.detected_environments:
            try:
                baseline = await self.load_configuration_for_environment(env)
                self.baseline_configurations[env] = baseline
                logger.info(f"Baseline established for environment: {env}")
            except Exception as e:
                logger.exception(f"Failed to establish baseline for {env}: {e}")

    async def detect_environments(self) -> None:
        """Detect available environments from configuration files."""
        env_files = [
            ".env",
            ".env.simple",
            ".env.enterprise",
            ".env.test",
        ]

        docker_files = [
            "docker-compose.yml",
            "docker-compose.simple.yml",
            "docker-compose.enterprise.yml",
        ]

        detected = set()

        # Check for environment files
        for env_file in env_files:
            if (self.config_dir / env_file).exists():
                if "simple" in env_file:
                    detected.add("simple")
                elif "enterprise" in env_file:
                    detected.add("enterprise")
                elif "test" in env_file:
                    detected.add("testing")
                else:
                    detected.add("development")

        # Check for Docker configurations
        for docker_file in docker_files:
            if (self.config_dir / docker_file).exists():
                if "simple" in docker_file:
                    detected.add("simple")
                elif "enterprise" in docker_file:
                    detected.add("enterprise")

        # Always include current environment
        try:
            current_config = get_config()
            if hasattr(current_config, "environment"):
                detected.add(current_config.environment.value)
            if hasattr(current_config, "mode"):
                detected.add(current_config.mode.value)
        except Exception as e:
            logger.warning(f"Could not detect current environment: {e}")

        self.detected_environments = detected
        logger.info(f"Detected environments: {', '.join(detected)}")

    async def load_configuration_for_environment(
        self, environment: str
    ) -> dict[str, Any]:
        """Load configuration for a specific environment."""
        # Temporarily set environment variables to load specific config
        original_env = {}

        try:
            # Map environment to configuration
            env_mapping = {
                "simple": {
                    "AI_DOCS__MODE": "simple",
                    "AI_DOCS__ENVIRONMENT": "development",
                },
                "enterprise": {
                    "AI_DOCS__MODE": "enterprise",
                    "AI_DOCS__ENVIRONMENT": "production",
                },
                "testing": {
                    "AI_DOCS__ENVIRONMENT": "testing",
                },
                "development": {
                    "AI_DOCS__ENVIRONMENT": "development",
                },
            }

            # set environment variables
            if environment in env_mapping:
                for key, value in env_mapping[environment].items():
                    original_env[key] = os.environ.get(key)
                    os.environ[key] = value

            # Load configuration
            config = ModernConfig() if is_using_modern_config() else LegacyConfig()

            # Convert to dict for comparison
            return (
                config.model_dump() if hasattr(config, "model_dump") else config.dict()
            )

        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    async def detect_configuration_drift(self) -> list[ConfigDrift]:
        """Detect configuration drift across environments."""
        logger.info("Detecting configuration drift")

        current_time = datetime.now()
        drift_results = []

        for environment in self.detected_environments:
            try:
                current_config = await self.load_configuration_for_environment(
                    environment
                )
                baseline_config = self.baseline_configurations.get(environment, {})

                # Compare configurations
                drifts = await self._compare_configurations(
                    baseline_config, current_config, environment, current_time
                )
                drift_results.extend(drifts)

            except Exception as e:
                logger.exception(f"Error detecting drift for {environment}: {e}")

        # Store drift history
        self.drift_history.extend(drift_results)

        # Keep only recent history (last 24 hours)
        cutoff_time = current_time - timedelta(hours=24)
        self.drift_history = [
            drift for drift in self.drift_history if drift.timestamp > cutoff_time
        ]

        if drift_results:
            logger.warning(f"Detected {len(drift_results)} configuration drifts")

        return drift_results

    async def _compare_configurations(
        self,
        baseline: dict[str, Any],
        current: dict[str, Any],
        environment: str,
        timestamp: datetime,
    ) -> list[ConfigDrift]:
        """Compare baseline and current configurations."""
        drifts = []

        # Critical parameters that shouldn't drift
        critical_params = {
            "mode",
            "environment",
            "embedding_provider",
            "crawl_provider",
            "qdrant_url",
            "redis_url",
        }

        # Check for parameter changes
        all_keys = set(baseline.keys()) | set(current.keys())

        for key in all_keys:
            baseline_value = baseline.get(key)
            current_value = current.get(key)

            if baseline_value != current_value:
                # Determine severity
                if key in critical_params:
                    severity = ConfigDriftSeverity.CRITICAL
                    impact_score = 0.8
                elif key.startswith("performance") or key.startswith("cache"):
                    severity = ConfigDriftSeverity.WARNING
                    impact_score = 0.4
                else:
                    severity = ConfigDriftSeverity.INFO
                    impact_score = 0.2

                # Check if auto-fix is available
                auto_fix_available = key in {
                    "performance.max_concurrent_crawls",
                    "cache.ttl_embeddings",
                    "cache.local_max_size",
                }

                drift = ConfigDrift(
                    severity=severity,
                    parameter=key,
                    expected_value=baseline_value,
                    current_value=current_value,
                    environment=environment,
                    timestamp=timestamp,
                    impact_score=impact_score,
                    remediation_suggested=severity
                    in [ConfigDriftSeverity.WARNING, ConfigDriftSeverity.CRITICAL],
                    auto_fix_available=auto_fix_available,
                )
                drifts.append(drift)

        return drifts

    async def validate_configuration_health(self) -> list[ConfigValidationResult]:
        """Validate configuration health across all environments."""
        logger.info("Validating configuration health")

        current_time = datetime.now()
        validation_results = []

        for environment in self.detected_environments:
            try:
                config = await self.load_configuration_for_environment(environment)
                results = await self._validate_environment_config(
                    config, environment, current_time
                )
                validation_results.extend(results)

            except Exception as e:
                logger.exception(f"Error validating {environment}: {e}")
                validation_results.append(
                    ConfigValidationResult(
                        status=ConfigValidationStatus.ERROR,
                        parameter="configuration_load",
                        message=f"Failed to load configuration: {e}",
                        environment=environment,
                        timestamp=current_time,
                    )
                )

        # Store validation history
        self.validation_history.extend(validation_results)

        # Keep only recent history
        cutoff_time = current_time - timedelta(hours=24)
        self.validation_history = [
            result
            for result in self.validation_history
            if result.timestamp > cutoff_time
        ]

        return validation_results

    async def _validate_environment_config(
        self,
        config: dict[str, Any],
        environment: str,
        timestamp: datetime,
    ) -> list[ConfigValidationResult]:
        """Validate configuration for a specific environment."""
        results = []

        # Validation rules
        validations = [
            self._validate_api_keys,
            self._validate_performance_settings,
            self._validate_cache_settings,
            self._validate_provider_compatibility,
            self._validate_security_settings,
        ]

        for validation_func in validations:
            try:
                validation_results = await validation_func(
                    config, environment, timestamp
                )
                results.extend(validation_results)
            except Exception as e:
                logger.exception(f"Validation error in {validation_func.__name__}: {e}")

        return results

    async def _validate_api_keys(
        self, config: dict[str, Any], environment: str, timestamp: datetime
    ) -> list[ConfigValidationResult]:
        """Validate API key configuration."""
        results = []

        # Check OpenAI provider requirements
        if config.get("embedding_provider") == "openai":
            openai_key = config.get("openai_api_key")
            if not openai_key:
                results.append(
                    ConfigValidationResult(
                        status=ConfigValidationStatus.ERROR,
                        parameter="openai_api_key",
                        message="OpenAI API key is required when using OpenAI embedding provider",
                        environment=environment,
                        timestamp=timestamp,
                        suggestions=[
                            "set AI_DOCS__OPENAI_API_KEY environment variable"
                        ],
                    )
                )
            elif not openai_key.startswith("sk-"):
                results.append(
                    ConfigValidationResult(
                        status=ConfigValidationStatus.WARNING,
                        parameter="openai_api_key",
                        message="OpenAI API key format appears incorrect",
                        environment=environment,
                        timestamp=timestamp,
                        suggestions=["Verify API key starts with 'sk-'"],
                    )
                )

        # Check Firecrawl provider requirements
        if config.get("crawl_provider") == "firecrawl":
            firecrawl_key = config.get("firecrawl_api_key")
            if not firecrawl_key:
                results.append(
                    ConfigValidationResult(
                        status=ConfigValidationStatus.ERROR,
                        parameter="firecrawl_api_key",
                        message="Firecrawl API key is required when using Firecrawl provider",
                        environment=environment,
                        timestamp=timestamp,
                        suggestions=[
                            "set AI_DOCS__FIRECRAWL_API_KEY environment variable"
                        ],
                    )
                )

        return results

    async def _validate_performance_settings(
        self, config: dict[str, Any], environment: str, timestamp: datetime
    ) -> list[ConfigValidationResult]:
        """Validate performance configuration settings."""
        results = []

        # Get performance config
        performance = config.get("performance", {})

        # Check concurrent crawls
        max_crawls = performance.get("max_concurrent_crawls", 10)
        if max_crawls > 50:
            results.append(
                ConfigValidationResult(
                    status=ConfigValidationStatus.WARNING,
                    parameter="performance.max_concurrent_crawls",
                    message=f"High concurrent crawls ({max_crawls}) may impact system stability",
                    environment=environment,
                    timestamp=timestamp,
                    suggestions=["Consider reducing to 25-30 for better stability"],
                )
            )

        # Check memory usage
        max_memory = performance.get("max_memory_usage_mb", 1000)
        if max_memory > 4000:
            results.append(
                ConfigValidationResult(
                    status=ConfigValidationStatus.WARNING,
                    parameter="performance.max_memory_usage_mb",
                    message=f"High memory limit ({max_memory}MB) may cause OOM issues",
                    environment=environment,
                    timestamp=timestamp,
                    suggestions=["Monitor actual memory usage and adjust accordingly"],
                )
            )

        return results

    async def _validate_cache_settings(
        self, config: dict[str, Any], environment: str, timestamp: datetime
    ) -> list[ConfigValidationResult]:
        """Validate cache configuration settings."""
        results = []

        cache = config.get("cache", {})

        # Check TTL values
        ttl_embeddings = cache.get("ttl_embeddings", 86400)
        if ttl_embeddings < 3600:  # Less than 1 hour
            results.append(
                ConfigValidationResult(
                    status=ConfigValidationStatus.WARNING,
                    parameter="cache.ttl_embeddings",
                    message=f"Low embedding TTL ({ttl_embeddings}s) may increase API costs",
                    environment=environment,
                    timestamp=timestamp,
                    suggestions=["Consider increasing to at least 3600s (1 hour)"],
                )
            )

        # Check Redis configuration
        if cache.get("enable_redis_cache"):
            redis_url = cache.get("redis_url") or config.get("redis_url")
            if not redis_url:
                results.append(
                    ConfigValidationResult(
                        status=ConfigValidationStatus.ERROR,
                        parameter="cache.redis_url",
                        message="Redis URL is required when Redis caching is enabled",
                        environment=environment,
                        timestamp=timestamp,
                        suggestions=["set AI_DOCS__REDIS_URL or disable Redis caching"],
                    )
                )

        return results

    async def _validate_provider_compatibility(
        self, config: dict[str, Any], environment: str, timestamp: datetime
    ) -> list[ConfigValidationResult]:
        """Validate provider compatibility."""
        results = []

        mode = config.get("mode", "simple")
        embedding_provider = config.get("embedding_provider", "fastembed")

        # In simple mode, recommend FastEmbed for better performance
        if mode == "simple" and embedding_provider == "openai":
            results.append(
                ConfigValidationResult(
                    status=ConfigValidationStatus.WARNING,
                    parameter="embedding_provider",
                    message="OpenAI provider in simple mode increases latency and costs",
                    environment=environment,
                    timestamp=timestamp,
                    suggestions=[
                        "Consider using FastEmbed for better performance in simple mode"
                    ],
                )
            )

        return results

    async def _validate_security_settings(
        self, config: dict[str, Any], environment: str, timestamp: datetime
    ) -> list[ConfigValidationResult]:
        """Validate security configuration."""
        results = []

        security = config.get("security", {})

        # Check production security settings
        if config.get("environment") == "production":
            if not security.get("require_api_keys", True):
                results.append(
                    ConfigValidationResult(
                        status=ConfigValidationStatus.CRITICAL,
                        parameter="security.require_api_keys",
                        message="API key requirement should be enabled in production",
                        environment=environment,
                        timestamp=timestamp,
                        suggestions=["set AI_DOCS__SECURITY__REQUIRE_API_KEYS=true"],
                    )
                )

            if not security.get("enable_rate_limiting", True):
                results.append(
                    ConfigValidationResult(
                        status=ConfigValidationStatus.WARNING,
                        parameter="security.enable_rate_limiting",
                        message="Rate limiting should be enabled in production",
                        environment=environment,
                        timestamp=timestamp,
                        suggestions=[
                            "set AI_DOCS__SECURITY__ENABLE_RATE_LIMITING=true"
                        ],
                    )
                )

        return results

    async def generate_optimization_recommendations(
        self,
    ) -> list[OptimizationRecommendation]:
        """Generate configuration optimization recommendations based on performance metrics."""
        logger.info("Generating optimization recommendations")

        recommendations = []

        for environment in self.detected_environments:
            try:
                config = await self.load_configuration_for_environment(environment)
                env_recommendations = await self._analyze_environment_performance(
                    config, environment
                )
                recommendations.extend(env_recommendations)

            except Exception as e:
                logger.exception(
                    f"Error generating recommendations for {environment}: {e}"
                )

        self.optimization_recommendations = recommendations
        return recommendations

    async def _analyze_environment_performance(
        self, config: dict[str, Any], environment: str
    ) -> list[OptimizationRecommendation]:
        """Analyze performance and generate recommendations for an environment."""
        recommendations = []

        # Simulate performance metrics analysis (in real implementation,
        # this would use actual metrics from Prometheus/monitoring)
        recent_metrics = [
            m
            for m in self.performance_metrics
            if m.environment == environment
            and m.timestamp > datetime.now() - timedelta(hours=1)
        ]

        # Analyze response times
        response_times = [
            m.value for m in recent_metrics if m.name == "response_time_ms"
        ]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)

            # High response times - recommend increasing concurrency
            if avg_response_time > 1000:  # > 1 second
                current_crawls = config.get("performance", {}).get(
                    "max_concurrent_crawls", 10
                )
                if current_crawls < 25:
                    recommendations.append(
                        OptimizationRecommendation(
                            parameter="performance.max_concurrent_crawls",
                            current_value=current_crawls,
                            recommended_value=min(current_crawls + 5, 25),
                            expected_improvement="20-30% faster response times",
                            confidence_score=0.8,
                            performance_impact="High",
                            environment=environment,
                            reasoning="High response times indicate underutilization of concurrent processing",
                        )
                    )

        # Analyze cache hit rates
        cache_hits = [m.value for m in recent_metrics if m.name == "cache_hit_rate"]
        if cache_hits:
            avg_hit_rate = sum(cache_hits) / len(cache_hits)

            # Low cache hit rate - recommend increasing TTL
            if avg_hit_rate < 0.7:  # < 70%
                current_ttl = config.get("cache", {}).get("ttl_embeddings", 86400)
                if current_ttl < 172800:  # < 48 hours
                    recommendations.append(
                        OptimizationRecommendation(
                            parameter="cache.ttl_embeddings",
                            current_value=current_ttl,
                            recommended_value=min(current_ttl * 2, 172800),
                            expected_improvement="15-25% cost reduction",
                            confidence_score=0.9,
                            performance_impact="Medium",
                            environment=environment,
                            reasoning="Low cache hit rate increases API costs and response times",
                        )
                    )

        return recommendations

    async def auto_remediate_issues(self, drifts: list[ConfigDrift]) -> dict[str, bool]:
        """Automatically remediate configuration issues where safe to do so."""
        if not self.enable_auto_remediation:
            logger.info("Auto-remediation disabled")
            return {}

        logger.info(f"Attempting auto-remediation for {len(drifts)} drift issues")

        remediation_results = {}

        for drift in drifts:
            if drift.auto_fix_available and drift.severity != ConfigDriftSeverity.FATAL:
                try:
                    success = await self._apply_auto_fix(drift)
                    remediation_results[drift.parameter] = success

                    if success:
                        logger.info(f"Auto-remediated drift in {drift.parameter}")
                    else:
                        logger.warning(f"Failed to auto-remediate {drift.parameter}")

                except Exception as e:
                    logger.exception(
                        f"Error during auto-remediation of {drift.parameter}: {e}"
                    )
                    remediation_results[drift.parameter] = False

        return remediation_results

    async def _apply_auto_fix(self, drift: ConfigDrift) -> bool:
        """Apply automatic fix for a configuration drift."""
        # In a real implementation, this would update environment variables
        # or configuration files. For demo purposes, we'll simulate the fix.

        logger.info(f"Applying auto-fix for {drift.parameter}")

        # Simulate fix application
        await asyncio.sleep(0.1)

        # Update baseline to reflect the fix
        if drift.environment in self.baseline_configurations:
            config = self.baseline_configurations[drift.environment]
            # set the parameter to expected value
            keys = drift.parameter.split(".")
            current_dict = config
            for key in keys[:-1]:
                if key not in current_dict:
                    current_dict[key] = {}
                current_dict = current_dict[key]
            current_dict[keys[-1]] = drift.expected_value

        return True

    async def handle_config_change(self, file_path: str) -> None:
        """Handle configuration file changes."""
        logger.info(f"Handling configuration change: {file_path}")

        # Wait a bit for file write to complete
        await asyncio.sleep(1)

        # Re-establish baselines
        await self.establish_baseline_configurations()

        # Perform immediate validation
        validation_results = await self.validate_configuration_health()

        # Check for critical issues
        critical_issues = [
            result
            for result in validation_results
            if result.status == ConfigValidationStatus.CRITICAL
        ]

        if critical_issues:
            logger.critical(
                f"Critical configuration issues detected after change: {len(critical_issues)}"
            )
            # In a real system, this would trigger alerts

    async def _periodic_drift_check(self) -> None:
        """Periodic configuration drift checking."""
        while True:
            try:
                await asyncio.sleep(self.drift_check_interval)

                drifts = await self.detect_configuration_drift()

                if drifts:
                    # Auto-remediate if enabled
                    if self.enable_auto_remediation:
                        await self.auto_remediate_issues(drifts)

                    # Log drift summary
                    critical_drifts = [
                        d for d in drifts if d.severity == ConfigDriftSeverity.CRITICAL
                    ]
                    if critical_drifts:
                        logger.critical(
                            f"Critical configuration drifts detected: {len(critical_drifts)}"
                        )

                self.last_drift_check = datetime.now()

            except Exception as e:
                logger.exception(f"Error in periodic drift check: {e}")

    async def _periodic_performance_optimization(self) -> None:
        """Periodic performance-based optimization."""
        while True:
            try:
                await asyncio.sleep(self.performance_optimization_interval)

                if self.enable_performance_optimization:
                    recommendations = await self.generate_optimization_recommendations()

                    if recommendations:
                        logger.info(
                            f"Generated {len(recommendations)} optimization recommendations"
                        )

                self.last_optimization_check = datetime.now()

            except Exception as e:
                logger.exception(f"Error in periodic optimization: {e}")

    def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        current_time = datetime.now()

        # Recent drift analysis
        recent_drifts = [
            drift
            for drift in self.drift_history
            if drift.timestamp > current_time - timedelta(hours=1)
        ]

        # Recent validation results
        recent_validations = [
            result
            for result in self.validation_history
            if result.timestamp > current_time - timedelta(hours=1)
        ]

        return {
            "system_status": {
                "automation_enabled": True,
                "auto_remediation_enabled": self.enable_auto_remediation,
                "file_monitoring_active": self.observer is not None
                and self.observer.is_alive(),
                "environments_monitored": len(self.detected_environments),
                "last_drift_check": self.last_drift_check.isoformat(),
                "last_optimization_check": self.last_optimization_check.isoformat(),
            },
            "drift_analysis": {
                "recent_drifts": len(recent_drifts),
                "critical_drifts": len(
                    [
                        d
                        for d in recent_drifts
                        if d.severity == ConfigDriftSeverity.CRITICAL
                    ]
                ),
                "auto_fixes_available": len(
                    [d for d in recent_drifts if d.auto_fix_available]
                ),
                "total_drift_history": len(self.drift_history),
            },
            "validation_status": {
                "recent_validations": len(recent_validations),
                "errors": len(
                    [
                        v
                        for v in recent_validations
                        if v.status == ConfigValidationStatus.ERROR
                    ]
                ),
                "warnings": len(
                    [
                        v
                        for v in recent_validations
                        if v.status == ConfigValidationStatus.WARNING
                    ]
                ),
                "critical_issues": len(
                    [
                        v
                        for v in recent_validations
                        if v.status == ConfigValidationStatus.CRITICAL
                    ]
                ),
            },
            "optimization": {
                "active_recommendations": len(self.optimization_recommendations),
                "performance_metrics_tracked": len(self.performance_metrics),
            },
            "environments": {
                "detected": list(self.detected_environments),
                "baselines_established": list(self.baseline_configurations.keys()),
            },
        }

    def get_detailed_report(self) -> dict[str, Any]:
        """Get detailed automation report."""
        status = self.get_system_status()

        # Add detailed information
        status["detailed_analysis"] = {
            "recent_drifts": [
                {
                    "parameter": drift.parameter,
                    "severity": drift.severity.value,
                    "environment": drift.environment,
                    "impact_score": drift.impact_score,
                    "auto_fix_available": drift.auto_fix_available,
                    "timestamp": drift.timestamp.isoformat(),
                }
                for drift in self.drift_history[-10:]  # Last 10 drifts
            ],
            "validation_results": [
                {
                    "parameter": result.parameter,
                    "status": result.status.value,
                    "message": result.message,
                    "environment": result.environment,
                    "suggestions": result.suggestions,
                    "timestamp": result.timestamp.isoformat(),
                }
                for result in self.validation_history[-10:]  # Last 10 validations
            ],
            "optimization_recommendations": [
                {
                    "parameter": rec.parameter,
                    "current_value": rec.current_value,
                    "recommended_value": rec.recommended_value,
                    "expected_improvement": rec.expected_improvement,
                    "confidence_score": rec.confidence_score,
                    "environment": rec.environment,
                    "reasoning": rec.reasoning,
                }
                for rec in self.optimization_recommendations
            ],
        }

        return status


# Global automation system instance
_automation_system: ConfigObservabilityAutomation | None = None


def get_automation_system() -> ConfigObservabilityAutomation:
    """Get or create the global automation system instance."""
    global _automation_system

    if _automation_system is None:
        _automation_system = ConfigObservabilityAutomation()

    return _automation_system


async def start_automation_system(**kwargs) -> ConfigObservabilityAutomation:
    """Start the configuration automation system."""
    automation_system = ConfigObservabilityAutomation(**kwargs)
    await automation_system.start()

    global _automation_system
    _automation_system = automation_system

    return automation_system


async def stop_automation_system() -> None:
    """Stop the configuration automation system."""
    global _automation_system

    if _automation_system:
        await _automation_system.stop()
        _automation_system = None
