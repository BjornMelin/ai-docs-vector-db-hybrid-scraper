"""Property-based tests for configuration state transitions.

Uses Hypothesis to generate various configuration states and transitions,
ensuring the system maintains consistency and correctness across all
possible state changes.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest
from hypothesis import assume, given, note, strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)
from pydantic import ValidationError

from src.config.core import Config
from src.config.drift_detection import (
    ConfigDriftDetector,
    DriftDetectionConfig,
    DriftSeverity,
    DriftType,
)
from src.config.enums import CrawlProvider, EmbeddingProvider, Environment, LogLevel
from src.config.reload import ConfigReloader
from src.config.security import (
    ConfigDataClassification,
    SecureConfigManager,
    SecurityConfig,
)


# Custom Hypothesis strategies for configuration data
def valid_api_key(prefix: str = "sk-") -> st.SearchStrategy[str]:
    """Generate valid API keys."""
    return st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        min_size=20,
        max_size=50,
    ).map(lambda s: f"{prefix}{s}")


def valid_url() -> st.SearchStrategy[str]:
    """Generate valid URLs."""
    return st.sampled_from(
        [
            "http://localhost:6333",
            "https://api.example.com",
            "http://127.0.0.1:8080",
            "https://test.service.io",
        ]
    )


def config_dict_strategy() -> st.SearchStrategy[dict[str, Any]]:
    """Generate valid configuration dictionaries."""
    return st.fixed_dictionaries(
        {
            "environment": st.sampled_from([e.value for e in Environment]),
            "log_level": st.sampled_from([l.value for l in LogLevel]),
            "embedding_provider": st.sampled_from([p.value for p in EmbeddingProvider]),
            "crawl_provider": st.sampled_from([p.value for p in CrawlProvider]),
            "api_base_url": valid_url(),
            "openai_api_key": st.one_of(st.none(), valid_api_key("sk-")),
            "qdrant_url": valid_url(),
            "cache_ttl_seconds": st.integers(min_value=60, max_value=86400),
            "max_concurrent_requests": st.integers(min_value=1, max_value=100),
        }
    )


class ConfigurationStateMachine(RuleBasedStateMachine):
    """State machine for testing configuration transitions."""

    def __init__(self):
        super().__init__()
        self.temp_dir = Path("/tmp/config_state_test")
        self.temp_dir.mkdir(exist_ok=True)

        # Initialize components
        self.config_file = self.temp_dir / ".env"
        self.config_file.write_text("ENVIRONMENT=testing")

        self.reloader = ConfigReloader(
            config_source=self.config_file,
            enable_signal_handler=False,
        )
        self.security_config = SecurityConfig()
        self.secure_manager = SecureConfigManager(
            self.security_config,
            config_dir=self.temp_dir,
        )
        self.drift_config = DriftDetectionConfig(
            snapshot_interval_minutes=1,
            comparison_interval_minutes=1,
        )
        self.drift_detector = ConfigDriftDetector(self.drift_config)

        # State tracking
        self.current_config: Config | None = None
        self.config_history: list[Config] = []
        self.reload_count = 0
        self.encryption_enabled = False
        self.listeners_added = 0

    configs = Bundle("configs")
    encrypted_items = Bundle("encrypted_items")
    listeners = Bundle("listeners")

    @initialize(target=configs)
    def init_config(self):
        """Initialize with a valid configuration."""
        config = Config()
        self.current_config = config
        self.config_history.append(config)
        self.reloader.set_current_config(config)
        return config

    @rule(
        target=configs,
        config_data=config_dict_strategy(),
    )
    def create_config(self, config_data: dict[str, Any]):
        """Create a new configuration from data."""
        try:
            # Write config data to file
            env_content = "\n".join(
                f"{k.upper()}={v}" for k, v in config_data.items() if v is not None
            )
            self.config_file.write_text(env_content)

            # Create config object
            config = Config.model_validate(config_data)

            # Store in history
            self.config_history.append(config)
            note(f"Created config with environment={config.environment}")

            return config

        except ValidationError as e:
            # This is expected for some invalid combinations
            note(f"Config validation failed: {e}")
            assume(False)  # Skip this example

    @rule(config=configs)
    def reload_configuration(self, config: Config):
        """Reload configuration through the reloader."""
        # Set as current config
        self.reloader.set_current_config(config)

        # Perform reload
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.reloader.reload_config(force=True))

            if result.success:
                self.reload_count += 1
                self.current_config = config
                note(f"Reload #{self.reload_count} successful")
            else:
                note(f"Reload failed: {result.error_message}")

        finally:
            loop.close()

    @rule(
        config=configs,
        classification=st.sampled_from(list(ConfigDataClassification)),
    )
    def encrypt_configuration(
        self, config: Config, classification: ConfigDataClassification
    ):
        """Encrypt configuration data."""
        config_data = config.model_dump()

        # Encrypt sensitive fields
        sensitive_fields = ["openai_api_key", "anthropic_api_key", "firecrawl_api_key"]

        for field in sensitive_fields:
            if config_data.get(field):
                encrypted = self.secure_manager.encrypt_config_value(
                    config_data[field],
                    classification,
                )
                # Store reference
                note(f"Encrypted {field} with classification {classification.value}")
                self.encryption_enabled = True
                return encrypted

    @rule(
        name=st.text(min_size=5, max_size=20),
        priority=st.integers(min_value=0, max_value=100),
        async_callback=st.booleans(),
    )
    def add_change_listener(self, name: str, priority: int, async_callback: bool):
        """Add a configuration change listener."""

        def sync_listener(_old_cfg, _new_cfg):
            time.sleep(0.01)  # Simulate work
            return True

        async def async_listener(_old_cfg, _new_cfg):
            await asyncio.sleep(0.01)  # Simulate async work
            return True

        self.reloader.add_change_listener(
            name=f"listener_{name}",
            callback=async_listener if async_callback else sync_listener,
            priority=priority,
            async_callback=async_callback,
        )

        self.listeners_added += 1
        note(f"Added listener #{self.listeners_added}: {name}")
        return name

    @rule(config=configs)
    def detect_drift(self, config: Config):
        """Detect configuration drift."""
        if len(self.config_history) < 2:
            return

        # Compare with previous config
        old_config = self.config_history[-2]
        new_config = config

        # Create snapshots
        old_snapshot = self.drift_detector._create_snapshot(
            config_data=old_config.model_dump(),
            source="test",
        )
        new_snapshot = self.drift_detector._create_snapshot(
            config_data=new_config.model_dump(),
            source="test",
        )

        # Detect differences
        differences = self._find_config_differences(
            old_config.model_dump(),
            new_config.model_dump(),
        )

        if differences:
            note(f"Drift detected: {len(differences)} changes")

    @rule()
    def trigger_rollback(self):
        """Trigger configuration rollback."""
        if len(self.config_history) < 2:
            return

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.reloader.rollback_config())

            if result.success:
                # Rollback to previous config
                self.current_config = self.config_history[-2]
                note("Rollback successful")
            else:
                note(f"Rollback failed: {result.error_message}")

        finally:
            loop.close()

    @invariant()
    def config_consistency(self):
        """Ensure configuration remains consistent."""
        if self.current_config:
            # Config should be valid
            try:
                self.current_config.model_validate(self.current_config.model_dump())
            except ValidationError as e:
                raise AssertionError(f"Current config became invalid: {e}")

            # Required fields should be present
            assert self.current_config.environment is not None
            assert self.current_config.log_level is not None

    @invariant()
    def reload_history_bounded(self):
        """Ensure reload history doesn't grow unbounded."""
        history = self.reloader.get_reload_history(limit=1000)
        assert len(history) <= 100  # Should maintain history limit

    @invariant()
    def no_resource_leaks(self):
        """Ensure no resource leaks."""
        # Check listener count is reasonable
        assert self.listeners_added < 1000

        # Check config history is bounded
        assert len(self.config_history) < 1000

    def _find_config_differences(
        self,
        old_config: dict[str, Any],
        new_config: dict[str, Any],
    ) -> list[tuple[str, Any, Any]]:
        """Find differences between two config dictionaries."""
        differences = []

        all_keys = set(old_config.keys()) | set(new_config.keys())

        for key in all_keys:
            old_value = old_config.get(key)
            new_value = new_config.get(key)

            if old_value != new_value:
                differences.append((key, old_value, new_value))

        return differences

    def teardown(self):
        """Clean up after test."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


# Run the state machine tests
TestConfigTransitions = ConfigurationStateMachine.TestCase


class TestConfigurationTransitionProperties:
    """Property-based tests for specific configuration transitions."""

    @given(
        initial_env=st.sampled_from(list(Environment)),
        target_env=st.sampled_from(list(Environment)),
        intermediate_steps=st.integers(min_value=0, max_value=5),
    )
    def test_environment_transitions(
        self,
        initial_env: Environment,
        target_env: Environment,
        intermediate_steps: int,
    ):
        """Test transitions between different environments."""
        config = Config(environment=initial_env)

        # Generate intermediate environments
        all_envs = list(Environment)
        transitions = [initial_env]

        for _ in range(intermediate_steps):
            transitions.append(all_envs[hash(str(transitions)) % len(all_envs)])

        transitions.append(target_env)

        # Apply transitions
        for env in transitions[1:]:
            config.environment = env
            # Validate after each transition
            config.model_validate(config.model_dump())

        # Final state should match target
        assert config.environment == target_env

    @given(
        providers=st.lists(
            st.sampled_from(list(EmbeddingProvider)),
            min_size=2,
            max_size=5,
        ),
    )
    def test_embedding_provider_switches(self, providers: list[EmbeddingProvider]):
        """Test switching between embedding providers."""
        config = Config()

        for provider in providers:
            old_provider = config.embedding_provider
            config.embedding_provider = provider

            # Verify provider-specific requirements
            if provider == EmbeddingProvider.OPENAI:
                if not config.openai.api_key:
                    config.openai.api_key = "test-key"

            # Validate configuration
            try:
                config.model_validate(config.model_dump())
            except ValidationError as e:
                # Some provider switches might require additional config
                note(
                    f"Validation failed switching from {old_provider} to {provider}: {e}"
                )

    @given(
        enable_sequence=st.lists(st.booleans(), min_size=5, max_size=20),
    )
    def test_feature_toggle_sequences(self, enable_sequence: list[bool]):
        """Test sequences of feature toggles."""
        config = Config()

        features = [
            (
                "cache.enable_caching",
                lambda c, v: setattr(c.cache, "enable_caching", v),
            ),
            (
                "performance.enable_performance_monitoring",
                lambda c, v: setattr(c.performance, "enable_performance_monitoring", v),
            ),
            (
                "monitoring.enable_task_monitoring",
                lambda c, v: setattr(c.monitoring, "enable_task_monitoring", v),
            ),
            ("security.enabled", lambda c, v: setattr(c.security, "enabled", v)),
        ]

        for i, enable in enumerate(enable_sequence):
            feature_name, setter = features[i % len(features)]
            setter(config, enable)

            # Validate after each toggle
            config.model_validate(config.model_dump())

            note(f"Set {feature_name} = {enable}")

    @given(
        data=st.data(),
        num_transitions=st.integers(min_value=5, max_value=20),
    )
    def test_random_valid_transitions(self, data, num_transitions: int):
        """Test random valid configuration transitions."""
        config = Config()

        for _i in range(num_transitions):
            # Choose random aspect to modify
            aspect = data.draw(
                st.sampled_from(
                    [
                        "performance",
                        "cache",
                        "security",
                        "monitoring",
                    ]
                )
            )

            if aspect == "performance":
                config.performance.max_concurrent_requests = data.draw(
                    st.integers(min_value=1, max_value=100)
                )
            elif aspect == "cache":
                config.cache.ttl_seconds = data.draw(
                    st.integers(min_value=60, max_value=3600)
                )
            elif aspect == "security":
                config.security.max_request_size_mb = data.draw(
                    st.integers(min_value=1, max_value=100)
                )
            elif aspect == "monitoring":
                config.monitoring.metrics_export_interval = data.draw(
                    st.integers(min_value=10, max_value=300)
                )

            # Validate after each change
            config.model_validate(config.model_dump())

        # Final config should still be valid
        final_dict = config.model_dump()
        Config.model_validate(final_dict)

    @given(
        classification_changes=st.lists(
            st.sampled_from(list(ConfigDataClassification)),
            min_size=3,
            max_size=10,
        ),
    )
    @pytest.mark.asyncio
    async def test_security_classification_transitions(
        self,
        classification_changes: list[ConfigDataClassification],
        tmp_path,
    ):
        """Test transitions between different security classifications."""
        security_config = SecurityConfig()
        manager = SecureConfigManager(security_config, config_dir=tmp_path)

        test_data = {"api_key": "secret_value", "setting": "public_value"}

        # Track classification history
        classification_history = []

        for classification in classification_changes:
            # Store with new classification
            encrypted = manager.encrypt_config_value(
                json.dumps(test_data),
                classification,
            )

            classification_history.append(
                {
                    "classification": classification,
                    "key_version": encrypted.key_version,
                    "timestamp": encrypted.created_at,
                }
            )

            # Verify we can decrypt
            decrypted = manager.decrypt_config_value(encrypted)
            assert json.loads(decrypted) == test_data

            # Audit trail should reflect classification
            audit_events = manager.get_audit_trail(limit=1)
            if audit_events:
                assert audit_events[0].data_classification == classification

        # Verify classification transitions were tracked
        assert len(classification_history) == len(classification_changes)

    @given(
        drift_sequence=st.lists(
            st.tuples(
                st.sampled_from(list(DriftType)),
                st.sampled_from(list(DriftSeverity)),
            ),
            min_size=5,
            max_size=15,
        ),
    )
    def test_drift_severity_escalation(
        self,
        drift_sequence: list[tuple[DriftType, DriftSeverity]],
    ):
        """Test drift detection with severity escalation."""
        config = DriftDetectionConfig()
        detector = ConfigDriftDetector(config)

        # Track severity progression
        severity_progression = []

        for drift_type, severity in drift_sequence:
            # Create drift event
            event = detector._create_drift_event(
                drift_type=drift_type,
                severity=severity,
                source="test",
                description=f"Test drift: {drift_type.value}",
                old_value="old",
                new_value="new",
            )

            severity_progression.append(severity)

            # Check if we should alert based on severity
            should_alert = severity in config.alert_on_severity

            if should_alert:
                note(
                    f"Alert triggered for {drift_type.value} with severity {severity.value}"
                )

        # Verify severity handling
        critical_count = severity_progression.count(DriftSeverity.CRITICAL)
        high_count = severity_progression.count(DriftSeverity.HIGH)

        # Critical and high severity should be rare
        assert critical_count <= len(drift_sequence) * 0.3
        assert high_count <= len(drift_sequence) * 0.5
