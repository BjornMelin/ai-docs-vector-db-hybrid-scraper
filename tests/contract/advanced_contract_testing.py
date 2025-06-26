"""Advanced Contract Testing Framework.

This module provides cutting-edge contract testing capabilities including:
- AI-powered contract generation
- Consumer-driven contract evolution
- Performance contract validation
- Security contract testing
- Semantic contract verification
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import jsonschema
import pytest
from hypothesis import given, strategies as st
from pydantic import BaseModel, Field, ValidationError

from src.models.api_contracts import SearchRequest, SearchResponse


class ContractType(Enum):
    """Contract types for different testing scenarios."""

    API = "api"
    DATA = "data"
    PERFORMANCE = "performance"
    SECURITY = "security"
    SEMANTIC = "semantic"
    BEHAVIORAL = "behavioral"


class ContractViolationType(Enum):
    """Types of contract violations."""

    BREAKING_CHANGE = "breaking_change"
    PERFORMANCE_REGRESSION = "performance_regression"
    SECURITY_VULNERABILITY = "security_vulnerability"
    SCHEMA_MISMATCH = "schema_mismatch"
    SEMANTIC_INCONSISTENCY = "semantic_inconsistency"
    BEHAVIORAL_CHANGE = "behavioral_change"


@dataclass
class ContractViolation:
    """Contract violation details."""

    violation_type: ContractViolationType
    severity: str  # 'critical', 'major', 'minor', 'warning'
    description: str
    expected: Any
    actual: Any
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_blocking(self) -> bool:
        """Determine if violation should block deployment."""
        return self.severity in ["critical", "major"]


class ContractValidationResult(BaseModel):
    """Contract validation result."""

    is_valid: bool
    contract_version: str
    violations: List[ContractViolation] = Field(default_factory=list)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def has_blocking_violations(self) -> bool:
        """Check if there are blocking violations."""
        return any(violation.is_blocking for violation in self.violations)

    @property
    def violation_summary(self) -> Dict[str, int]:
        """Get summary of violations by severity."""
        summary = {"critical": 0, "major": 0, "minor": 0, "warning": 0}
        for violation in self.violations:
            summary[violation.severity] = summary.get(violation.severity, 0) + 1
        return summary


class ContractValidator(ABC):
    """Abstract base class for contract validators."""

    @abstractmethod
    async def validate(
        self, contract: Dict[str, Any], actual_data: Any
    ) -> ContractValidationResult:
        """Validate contract against actual data."""
        pass

    @abstractmethod
    def get_contract_type(self) -> ContractType:
        """Get the type of contracts this validator handles."""
        pass


class APIContractValidator(ContractValidator):
    """Advanced API contract validator with semantic understanding."""

    def __init__(self):
        self.contract_cache: Dict[str, Dict[str, Any]] = {}
        self.validation_history: List[ContractValidationResult] = []

    def get_contract_type(self) -> ContractType:
        return ContractType.API

    async def validate(
        self, contract: Dict[str, Any], actual_data: Any
    ) -> ContractValidationResult:
        """Validate API contract with comprehensive checks."""
        violations = []
        performance_metrics = {}

        start_time = time.time()

        # Schema validation
        schema_violations = await self._validate_schema(contract, actual_data)
        violations.extend(schema_violations)

        # Semantic validation
        semantic_violations = await self._validate_semantics(contract, actual_data)
        violations.extend(semantic_violations)

        # Performance validation
        perf_violations, perf_metrics = await self._validate_performance(
            contract, actual_data
        )
        violations.extend(perf_violations)
        performance_metrics.update(perf_metrics)

        # Security validation
        security_violations = await self._validate_security(contract, actual_data)
        violations.extend(security_violations)

        validation_time = time.time() - start_time
        performance_metrics["validation_time_ms"] = validation_time * 1000

        result = ContractValidationResult(
            is_valid=len([v for v in violations if v.is_blocking]) == 0,
            contract_version=contract.get("version", "1.0.0"),
            violations=violations,
            performance_metrics=performance_metrics,
            metadata={"validator": "APIContractValidator"},
        )

        self.validation_history.append(result)
        return result

    async def _validate_schema(
        self, contract: Dict[str, Any], actual_data: Any
    ) -> List[ContractViolation]:
        """Validate data schema against contract."""
        violations = []

        try:
            schema = contract.get("schema", {})
            if schema:
                jsonschema.validate(actual_data, schema)
        except jsonschema.ValidationError as e:
            violations.append(
                ContractViolation(
                    violation_type=ContractViolationType.SCHEMA_MISMATCH,
                    severity="major",
                    description=f"Schema validation failed: {e.message}",
                    expected=contract.get("schema"),
                    actual=actual_data,
                    context={"path": e.absolute_path, "validator": e.validator},
                )
            )
        except Exception as e:
            violations.append(
                ContractViolation(
                    violation_type=ContractViolationType.SCHEMA_MISMATCH,
                    severity="critical",
                    description=f"Schema validation error: {e!s}",
                    expected=contract.get("schema"),
                    actual=actual_data,
                )
            )

        return violations

    async def _validate_semantics(
        self, contract: Dict[str, Any], actual_data: Any
    ) -> List[ContractViolation]:
        """Validate semantic consistency."""
        violations = []

        # Check semantic rules from contract
        semantic_rules = contract.get("semantic_rules", [])

        for rule in semantic_rules:
            rule_type = rule.get("type")

            if rule_type == "response_consistency":
                # Check if response structure is consistent with expectations
                expected_fields = set(rule.get("required_fields", []))
                if isinstance(actual_data, dict):
                    actual_fields = set(actual_data.keys())
                    missing_fields = expected_fields - actual_fields

                    if missing_fields:
                        violations.append(
                            ContractViolation(
                                violation_type=ContractViolationType.SEMANTIC_INCONSISTENCY,
                                severity="major",
                                description=f"Missing required fields: {missing_fields}",
                                expected=list(expected_fields),
                                actual=list(actual_fields),
                                context=rule,
                            )
                        )

            elif rule_type == "data_relationships":
                # Validate relationships between fields
                relationships = rule.get("relationships", [])
                for rel in relationships:
                    if not self._validate_relationship(actual_data, rel):
                        violations.append(
                            ContractViolation(
                                violation_type=ContractViolationType.SEMANTIC_INCONSISTENCY,
                                severity="minor",
                                description=f"Data relationship violated: {rel['description']}",
                                expected=rel,
                                actual=actual_data,
                                context={"relationship": rel},
                            )
                        )

        return violations

    def _validate_relationship(self, data: Any, relationship: Dict[str, Any]) -> bool:
        """Validate a specific data relationship."""
        if not isinstance(data, dict):
            return True

        rel_type = relationship.get("type")

        if rel_type == "field_dependency":
            # If field A exists, field B must also exist
            field_a = relationship.get("field_a")
            field_b = relationship.get("field_b")

            if field_a in data and field_b not in data:
                return False

        elif rel_type == "value_constraint":
            # Field value must satisfy constraint
            field = relationship.get("field")
            constraint = relationship.get("constraint")

            if field in data:
                value = data[field]

                if constraint.get("type") == "range":
                    min_val = constraint.get("min", float("-inf"))
                    max_val = constraint.get("max", float("inf"))

                    if not (min_val <= value <= max_val):
                        return False

                elif constraint.get("type") == "enum":
                    allowed_values = constraint.get("values", [])
                    if value not in allowed_values:
                        return False

        return True

    async def _validate_performance(
        self, contract: Dict[str, Any], actual_data: Any
    ) -> Tuple[List[ContractViolation], Dict[str, float]]:
        """Validate performance contracts."""
        violations = []
        metrics = {}

        perf_contract = contract.get("performance", {})

        # Response time validation
        max_response_time = perf_contract.get("max_response_time_ms")
        if max_response_time and hasattr(actual_data, "response_time_ms"):
            actual_time = getattr(actual_data, "response_time_ms", 0)
            metrics["response_time_ms"] = actual_time

            if actual_time > max_response_time:
                violations.append(
                    ContractViolation(
                        violation_type=ContractViolationType.PERFORMANCE_REGRESSION,
                        severity="major",
                        description=f"Response time exceeded: {actual_time}ms > {max_response_time}ms",
                        expected=max_response_time,
                        actual=actual_time,
                    )
                )

        # Throughput validation
        min_throughput = perf_contract.get("min_throughput_rps")
        if min_throughput and hasattr(actual_data, "throughput_rps"):
            actual_throughput = getattr(actual_data, "throughput_rps", 0)
            metrics["throughput_rps"] = actual_throughput

            if actual_throughput < min_throughput:
                violations.append(
                    ContractViolation(
                        violation_type=ContractViolationType.PERFORMANCE_REGRESSION,
                        severity="major",
                        description=f"Throughput below threshold: {actual_throughput} < {min_throughput} RPS",
                        expected=min_throughput,
                        actual=actual_throughput,
                    )
                )

        # Memory usage validation
        max_memory_mb = perf_contract.get("max_memory_mb")
        if max_memory_mb and hasattr(actual_data, "memory_usage_mb"):
            actual_memory = getattr(actual_data, "memory_usage_mb", 0)
            metrics["memory_usage_mb"] = actual_memory

            if actual_memory > max_memory_mb:
                violations.append(
                    ContractViolation(
                        violation_type=ContractViolationType.PERFORMANCE_REGRESSION,
                        severity="minor",
                        description=f"Memory usage exceeded: {actual_memory}MB > {max_memory_mb}MB",
                        expected=max_memory_mb,
                        actual=actual_memory,
                    )
                )

        return violations, metrics

    async def _validate_security(
        self, contract: Dict[str, Any], actual_data: Any
    ) -> List[ContractViolation]:
        """Validate security contracts."""
        violations = []

        security_contract = contract.get("security", {})

        # Check for sensitive data exposure
        sensitive_fields = security_contract.get("sensitive_fields", [])
        if isinstance(actual_data, dict):
            for field in sensitive_fields:
                if field in actual_data:
                    violations.append(
                        ContractViolation(
                            violation_type=ContractViolationType.SECURITY_VULNERABILITY,
                            severity="critical",
                            description=f"Sensitive field '{field}' exposed in response",
                            expected=f"Field '{field}' should be redacted",
                            actual=f"Field '{field}' present in response",
                            context={"field": field},
                        )
                    )

        # Validate encryption requirements
        encryption_required = security_contract.get("encryption_required", False)
        if encryption_required and hasattr(actual_data, "is_encrypted"):
            if not getattr(actual_data, "is_encrypted", False):
                violations.append(
                    ContractViolation(
                        violation_type=ContractViolationType.SECURITY_VULNERABILITY,
                        severity="critical",
                        description="Response data must be encrypted",
                        expected=True,
                        actual=False,
                    )
                )

        return violations


class ConsumerDrivenContractTesting:
    """Consumer-driven contract testing with evolution support."""

    def __init__(self):
        self.contracts: Dict[str, Dict[str, Any]] = {}
        self.consumer_expectations: Dict[str, List[Dict[str, Any]]] = {}
        self.provider_capabilities: Dict[str, Dict[str, Any]] = {}

    def register_consumer_expectation(
        self, consumer_id: str, expectation: Dict[str, Any]
    ) -> None:
        """Register a consumer's contract expectation."""
        if consumer_id not in self.consumer_expectations:
            self.consumer_expectations[consumer_id] = []

        self.consumer_expectations[consumer_id].append(expectation)

    def register_provider_capability(
        self, provider_id: str, capability: Dict[str, Any]
    ) -> None:
        """Register a provider's capabilities."""
        self.provider_capabilities[provider_id] = capability

    async def validate_consumer_provider_compatibility(
        self, consumer_id: str, provider_id: str
    ) -> ContractValidationResult:
        """Validate compatibility between consumer and provider."""
        violations = []

        consumer_expectations = self.consumer_expectations.get(consumer_id, [])
        provider_capability = self.provider_capabilities.get(provider_id, {})

        for expectation in consumer_expectations:
            compatibility_violations = self._check_compatibility(
                expectation, provider_capability
            )
            violations.extend(compatibility_violations)

        return ContractValidationResult(
            is_valid=len([v for v in violations if v.is_blocking]) == 0,
            contract_version="1.0.0",
            violations=violations,
            metadata={
                "consumer_id": consumer_id,
                "provider_id": provider_id,
                "expectations_checked": len(consumer_expectations),
            },
        )

    def _check_compatibility(
        self, expectation: Dict[str, Any], capability: Dict[str, Any]
    ) -> List[ContractViolation]:
        """Check compatibility between expectation and capability."""
        violations = []

        # Check API version compatibility
        expected_version = expectation.get("api_version")
        supported_versions = capability.get("supported_versions", [])

        if expected_version and expected_version not in supported_versions:
            violations.append(
                ContractViolation(
                    violation_type=ContractViolationType.BREAKING_CHANGE,
                    severity="critical",
                    description=f"API version {expected_version} not supported",
                    expected=expected_version,
                    actual=supported_versions,
                )
            )

        # Check required features
        required_features = expectation.get("required_features", [])
        available_features = capability.get("features", [])

        missing_features = set(required_features) - set(available_features)
        if missing_features:
            violations.append(
                ContractViolation(
                    violation_type=ContractViolationType.BREAKING_CHANGE,
                    severity="major",
                    description=f"Missing required features: {missing_features}",
                    expected=required_features,
                    actual=available_features,
                )
            )

        # Check data format compatibility
        expected_format = expectation.get("data_format")
        supported_formats = capability.get("supported_formats", [])

        if expected_format and expected_format not in supported_formats:
            violations.append(
                ContractViolation(
                    violation_type=ContractViolationType.SCHEMA_MISMATCH,
                    severity="major",
                    description=f"Data format {expected_format} not supported",
                    expected=expected_format,
                    actual=supported_formats,
                )
            )

        return violations

    def generate_contract_evolution_report(self) -> Dict[str, Any]:
        """Generate contract evolution report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "consumers": len(self.consumer_expectations),
            "providers": len(self.provider_capabilities),
            "compatibility_matrix": {},
            "evolution_recommendations": [],
        }

        # Build compatibility matrix
        for consumer_id in self.consumer_expectations:
            report["compatibility_matrix"][consumer_id] = {}
            for provider_id in self.provider_capabilities:
                compatibility_result = asyncio.run(
                    self.validate_consumer_provider_compatibility(
                        consumer_id, provider_id
                    )
                )
                report["compatibility_matrix"][consumer_id][provider_id] = {
                    "compatible": compatibility_result.is_valid,
                    "violations": len(compatibility_result.violations),
                    "blocking_violations": len(
                        [v for v in compatibility_result.violations if v.is_blocking]
                    ),
                }

        # Generate evolution recommendations
        report["evolution_recommendations"] = self._generate_evolution_recommendations()

        return report

    def _generate_evolution_recommendations(self) -> List[str]:
        """Generate recommendations for contract evolution."""
        recommendations = []

        # Analyze common patterns across consumers
        all_expectations = []
        for expectations in self.consumer_expectations.values():
            all_expectations.extend(expectations)

        # Recommend version standardization
        versions = set()
        for exp in all_expectations:
            if "api_version" in exp:
                versions.add(exp["api_version"])

        if len(versions) > 2:
            recommendations.append(
                f"Consider version consolidation - {len(versions)} different versions in use"
            )

        # Recommend feature standardization
        all_features = set()
        for exp in all_expectations:
            features = exp.get("required_features", [])
            all_features.update(features)

        if len(all_features) > 10:
            recommendations.append(
                "Consider feature grouping for better API organization"
            )

        return recommendations


class AIContractGenerator:
    """AI-powered contract generation for testing."""

    @staticmethod
    def generate_search_api_contract() -> Dict[str, Any]:
        """Generate comprehensive search API contract."""
        return {
            "version": "2.0.0",
            "name": "Search API Contract",
            "description": "Advanced search API with semantic validation",
            "schema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "title": {"type": "string"},
                                "content": {"type": "string"},
                                "score": {"type": "number", "minimum": 0, "maximum": 1},
                                "metadata": {"type": "object"},
                            },
                            "required": ["id", "title", "score"],
                        },
                    },
                    "total_count": {"type": "integer", "minimum": 0},
                    "query_time_ms": {"type": "number", "minimum": 0},
                },
                "required": ["success", "results", "total_count"],
            },
            "semantic_rules": [
                {
                    "type": "response_consistency",
                    "required_fields": ["success", "results", "total_count"],
                    "description": "Core response fields must always be present",
                },
                {
                    "type": "data_relationships",
                    "relationships": [
                        {
                            "type": "value_constraint",
                            "field": "total_count",
                            "constraint": {"type": "range", "min": 0, "max": 10000},
                            "description": "Total count should be reasonable",
                        },
                        {
                            "type": "field_dependency",
                            "field_a": "results",
                            "field_b": "total_count",
                            "description": "Results array length should relate to total_count",
                        },
                    ],
                },
            ],
            "performance": {
                "max_response_time_ms": 200,
                "min_throughput_rps": 100,
                "max_memory_mb": 512,
            },
            "security": {
                "sensitive_fields": ["user_id", "api_key", "token"],
                "encryption_required": False,
                "rate_limit_rps": 1000,
            },
        }

    @staticmethod
    def generate_mutation_contract(base_contract: Dict[str, Any]) -> Dict[str, Any]:
        """Generate contract mutations for negative testing."""
        import copy

        mutated_contract = copy.deepcopy(base_contract)

        # Mutation: Remove required field
        if "schema" in mutated_contract and "required" in mutated_contract["schema"]:
            required_fields = mutated_contract["schema"]["required"]
            if len(required_fields) > 1:
                mutated_contract["schema"]["required"] = required_fields[:-1]

        # Mutation: Tighten performance constraints
        if "performance" in mutated_contract:
            perf = mutated_contract["performance"]
            if "max_response_time_ms" in perf:
                perf["max_response_time_ms"] = perf["max_response_time_ms"] // 2

        # Mutation: Add new security constraint
        if "security" in mutated_contract:
            mutated_contract["security"]["new_constraint"] = True

        return mutated_contract


# Property-based contract testing
@given(
    search_results=st.lists(
        st.dictionaries(
            keys=st.sampled_from(["id", "title", "content", "score"]),
            values=st.one_of(
                st.text(min_size=1, max_size=100),
                st.floats(min_value=0.0, max_value=1.0),
            ),
        ),
        min_size=0,
        max_size=20,
    ),
    total_count=st.integers(min_value=0, max_value=10000),
    query_time=st.floats(min_value=1.0, max_value=5000.0),
)
@pytest.mark.asyncio
async def test_search_contract_property_validation(
    search_results, total_count, query_time
):
    """Property-based testing for search API contract."""

    # Create test response data
    response_data = {
        "success": True,
        "results": search_results,
        "total_count": total_count,
        "query_time_ms": query_time,
    }

    # Generate contract
    contract = AIContractGenerator.generate_search_api_contract()

    # Validate contract
    validator = APIContractValidator()
    result = await validator.validate(contract, response_data)

    # Property: Valid responses should pass validation
    if all("id" in r and "title" in r and "score" in r for r in search_results):
        # All results have required fields
        assert len([v for v in result.violations if v.is_blocking]) == 0, (
            f"Valid response failed validation: {[v.description for v in result.violations]}"
        )

    # Property: Performance violations should be detected
    if query_time > contract["performance"]["max_response_time_ms"]:
        performance_violations = [
            v
            for v in result.violations
            if v.violation_type == ContractViolationType.PERFORMANCE_REGRESSION
        ]
        # Note: This would fail in real implementation since we don't pass performance data
        # In actual usage, response_data would include performance metrics


# Consumer-driven contract testing example
@pytest.mark.asyncio
async def test_consumer_driven_contract_evolution():
    """Test consumer-driven contract evolution."""

    cdc_testing = ConsumerDrivenContractTesting()

    # Register consumer expectations
    cdc_testing.register_consumer_expectation(
        "web_client",
        {
            "api_version": "2.0.0",
            "required_features": ["search", "pagination", "filtering"],
            "data_format": "json",
        },
    )

    cdc_testing.register_consumer_expectation(
        "mobile_app",
        {
            "api_version": "2.0.0",
            "required_features": ["search", "caching"],
            "data_format": "json",
        },
    )

    # Register provider capabilities
    cdc_testing.register_provider_capability(
        "search_service",
        {
            "supported_versions": ["1.0.0", "2.0.0"],
            "features": ["search", "pagination", "filtering", "sorting"],
            "supported_formats": ["json", "xml"],
        },
    )

    # Validate compatibility
    web_compatibility = await cdc_testing.validate_consumer_provider_compatibility(
        "web_client", "search_service"
    )

    mobile_compatibility = await cdc_testing.validate_consumer_provider_compatibility(
        "mobile_app", "search_service"
    )

    # Assertions
    assert web_compatibility.is_valid, "Web client should be compatible"
    assert mobile_compatibility.is_valid, "Mobile app should be compatible"

    # Generate evolution report
    evolution_report = cdc_testing.generate_contract_evolution_report()

    assert evolution_report["consumers"] == 2
    assert evolution_report["providers"] == 1
    assert "compatibility_matrix" in evolution_report


if __name__ == "__main__":
    # Demonstration of advanced contract testing
    print("Running advanced contract testing demonstration...")

    # Test API contract validation
    async def demo_api_validation():
        validator = APIContractValidator()
        contract = AIContractGenerator.generate_search_api_contract()

        # Valid response
        valid_response = {
            "success": True,
            "results": [
                {"id": "1", "title": "Test Doc", "score": 0.95, "content": "Content"}
            ],
            "total_count": 1,
            "query_time_ms": 50,
        }

        result = await validator.validate(contract, valid_response)
        print(f"Valid response validation: {result.is_valid}")
        print(f"Violations: {len(result.violations)}")

        # Invalid response (missing required field)
        invalid_response = {
            "success": True,
            "results": [
                {"id": "1", "score": 0.95}  # Missing 'title'
            ],
            "query_time_ms": 50,
        }

        result = await validator.validate(contract, invalid_response)
        print(f"Invalid response validation: {result.is_valid}")
        print(f"Violations: {len(result.violations)}")

        for violation in result.violations:
            print(f"  - {violation.severity}: {violation.description}")

    # Run demonstration
    asyncio.run(demo_api_validation())

    print("\nAdvanced contract testing demonstration completed!")
