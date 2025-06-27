"""Data quality testing fixtures and configuration.

This module provides pytest fixtures for comprehensive data quality testing including
data validation, consistency checks, integrity verification, transformation testing,
and migration validation.
"""

import hashlib
import json
import random
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest


@dataclass
class DataQualityRule:
    """Data quality rule definition."""

    rule_id: str
    name: str
    description: str
    rule_type: (
        str  # "completeness", "validity", "consistency", "accuracy", "uniqueness"
    )
    severity: str  # "critical", "high", "medium", "low"
    field_name: str | None = None
    threshold: float | None = None
    pattern: str | None = None
    reference_data: dict[str, Any] | None = None


@dataclass
class DataQualityResult:
    """Data quality assessment result."""

    rule_id: str
    status: str  # "pass", "fail", "warning"
    score: float  # 0.0 to 1.0
    total_records: int
    failed_records: int
    passed_records: int
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DataIntegrityCheck:
    """Data integrity check configuration."""

    check_id: str
    name: str
    check_type: str  # "referential", "domain", "format", "range"
    table_name: str
    column_name: str
    constraint: dict[str, Any]
    severity: str = "high"
    auto_fix: bool = False


@pytest.fixture(scope="session")
def data_quality_config():
    """Provide data quality testing configuration."""
    return {
        "quality_dimensions": {
            "completeness": {
                "description": "Data is complete and not missing",
                "threshold": 0.95,
                "critical_fields": ["id", "created_at", "url"],
            },
            "validity": {
                "description": "Data conforms to defined formats and constraints",
                "threshold": 0.98,
                "validation_rules": ["format", "type", "range"],
            },
            "consistency": {
                "description": "Data is consistent across different sources/times",
                "threshold": 0.90,
                "check_types": ["cross_field", "temporal", "referential"],
            },
            "accuracy": {
                "description": "Data accurately represents real-world values",
                "threshold": 0.85,
                "verification_methods": ["sampling", "external_validation"],
            },
            "uniqueness": {
                "description": "Data does not contain duplicates",
                "threshold": 0.99,
                "key_fields": ["id", "url", "hash"],
            },
            "timeliness": {
                "description": "Data is current and up-to-date",
                "threshold": 0.90,
                "max_age_hours": 24,
            },
        },
        "sampling": {
            "default_sample_size": 1000,
            "max_sample_size": 10000,
            "stratified_sampling": True,
            "confidence_level": 0.95,
        },
        "profiling": {
            "enable_statistical_profiling": True,
            "enable_pattern_analysis": True,
            "enable_correlation_analysis": True,
            "max_unique_values": 100,
        },
        "remediation": {
            "auto_fix_enabled": False,
            "quarantine_bad_data": True,
            "notification_thresholds": {
                "critical": 0.05,
                "high": 0.10,
                "medium": 0.20,
            },
        },
    }


@pytest.fixture
def data_quality_validator():
    """Data quality validation utilities."""

    class DataQualityValidator:
        def __init__(self):
            self.rules = {}
            self.results = []
            self.data_profiles = {}

        def add_rule(self, rule: DataQualityRule):
            """Add a data quality rule."""
            self.rules[rule.rule_id] = rule

        def validate_completeness(
            self, data: list[dict[str, Any]], rule: DataQualityRule
        ) -> DataQualityResult:
            """Validate data completeness."""
            total_records = len(data)

            if rule.field_name:
                # Check specific field completeness
                failed_records = sum(
                    1
                    for record in data
                    if record.get(rule.field_name) is None
                    or record.get(rule.field_name) == ""
                )
            else:
                # Check overall record completeness
                failed_records = sum(
                    1 for record in data if not record or len(record) == 0
                )

            passed_records = total_records - failed_records
            score = passed_records / max(total_records, 1)

            status = "pass" if score >= (rule.threshold or 0.95) else "fail"

            return DataQualityResult(
                rule_id=rule.rule_id,
                status=status,
                score=score,
                total_records=total_records,
                failed_records=failed_records,
                passed_records=passed_records,
                details={
                    "field_name": rule.field_name,
                    "threshold": rule.threshold,
                    "completeness_rate": score,
                },
            )

        def validate_validity(
            self, data: list[dict[str, Any]], rule: DataQualityRule
        ) -> DataQualityResult:
            """Validate data validity (format, type, pattern)."""
            total_records = len(data)
            failed_records = 0
            validation_errors = []

            for record in data:
                if rule.field_name and rule.field_name in record:
                    value = record[rule.field_name]

                    # Type validation
                    if "expected_type" in rule.reference_data:
                        expected_type = rule.reference_data["expected_type"]
                        if not isinstance(value, expected_type):
                            failed_records += 1
                            validation_errors.append(
                                f"Type mismatch: expected {expected_type}, got {type(value)}"
                            )
                            continue

                    # Pattern validation
                    if rule.pattern and isinstance(value, str):
                        import re

                        if not re.match(rule.pattern, value):
                            failed_records += 1
                            validation_errors.append(
                                f"Pattern mismatch: {value} does not match {rule.pattern}"
                            )
                            continue

                    # Range validation
                    if (
                        "min_value" in rule.reference_data
                        or "max_value" in rule.reference_data
                    ) and isinstance(value, int | float):
                        min_val = rule.reference_data.get("min_value")
                        max_val = rule.reference_data.get("max_value")

                        if min_val is not None and value < min_val:
                            failed_records += 1
                            validation_errors.append(
                                f"Value {value} below minimum {min_val}"
                            )
                            continue

                        if max_val is not None and value > max_val:
                            failed_records += 1
                            validation_errors.append(
                                f"Value {value} above maximum {max_val}"
                            )
                            continue

            passed_records = total_records - failed_records
            score = passed_records / max(total_records, 1)
            status = "pass" if score >= (rule.threshold or 0.98) else "fail"

            return DataQualityResult(
                rule_id=rule.rule_id,
                status=status,
                score=score,
                total_records=total_records,
                failed_records=failed_records,
                passed_records=passed_records,
                details={
                    "field_name": rule.field_name,
                    "validation_type": rule.description,
                    "validity_rate": score,
                },
                errors=validation_errors[:10],  # Limit to first 10 errors
            )

        def validate_uniqueness(
            self, data: list[dict[str, Any]], rule: DataQualityRule
        ) -> DataQualityResult:
            """Validate data uniqueness."""
            total_records = len(data)

            if rule.field_name:
                # Check uniqueness of specific field
                values = [
                    record.get(rule.field_name)
                    for record in data
                    if rule.field_name in record
                ]
                unique_values = set(values)
                duplicates = len(values) - len(unique_values)
            else:
                # Check uniqueness of entire records
                record_hashes = []
                for record in data:
                    record_str = json.dumps(record, sort_keys=True)
                    record_hash = hashlib.sha256(record_str.encode()).hexdigest()
                    record_hashes.append(record_hash)

                unique_hashes = set(record_hashes)
                duplicates = len(record_hashes) - len(unique_hashes)

            failed_records = duplicates
            passed_records = total_records - failed_records
            score = passed_records / max(total_records, 1)
            status = "pass" if score >= (rule.threshold or 0.99) else "fail"

            return DataQualityResult(
                rule_id=rule.rule_id,
                status=status,
                score=score,
                total_records=total_records,
                failed_records=failed_records,
                passed_records=passed_records,
                details={
                    "field_name": rule.field_name,
                    "duplicate_count": duplicates,
                    "uniqueness_rate": score,
                },
            )

        def validate_consistency(
            self, data: list[dict[str, Any]], rule: DataQualityRule
        ) -> DataQualityResult:
            """Validate data consistency across fields or records."""
            total_records = len(data)
            failed_records = 0
            consistency_errors = []

            # Cross-field consistency checks
            if rule.reference_data and "consistency_rules" in rule.reference_data:
                for record in data:
                    for consistency_rule in rule.reference_data["consistency_rules"]:
                        rule_type = consistency_rule.get("type")

                        if rule_type == "field_relationship":
                            field1 = consistency_rule.get("field1")
                            field2 = consistency_rule.get("field2")
                            relationship = consistency_rule.get(
                                "relationship"
                            )  # "greater", "equal", "less"

                            if field1 in record and field2 in record:
                                val1 = record[field1]
                                val2 = record[field2]

                                if relationship == "greater" and val1 <= val2:
                                    failed_records += 1
                                    consistency_errors.append(
                                        f"{field1} ({val1}) should be greater than {field2} ({val2})"
                                    )
                                elif relationship == "equal" and val1 != val2:
                                    failed_records += 1
                                    consistency_errors.append(
                                        f"{field1} ({val1}) should equal {field2} ({val2})"
                                    )
                                elif relationship == "less" and val1 >= val2:
                                    failed_records += 1
                                    consistency_errors.append(
                                        f"{field1} ({val1}) should be less than {field2} ({val2})"
                                    )

            passed_records = total_records - failed_records
            score = passed_records / max(total_records, 1)
            status = "pass" if score >= (rule.threshold or 0.90) else "fail"

            return DataQualityResult(
                rule_id=rule.rule_id,
                status=status,
                score=score,
                total_records=total_records,
                failed_records=failed_records,
                passed_records=passed_records,
                details={
                    "consistency_type": rule.description,
                    "consistency_rate": score,
                },
                errors=consistency_errors[:10],
            )

        def run_all_validations(
            self, data: list[dict[str, Any]]
        ) -> list[DataQualityResult]:
            """Run all registered validation rules."""
            results = []

            for rule in self.rules.values():
                if rule.rule_type == "completeness":
                    result = self.validate_completeness(data, rule)
                elif rule.rule_type == "validity":
                    result = self.validate_validity(data, rule)
                elif rule.rule_type == "uniqueness":
                    result = self.validate_uniqueness(data, rule)
                elif rule.rule_type == "consistency":
                    result = self.validate_consistency(data, rule)
                else:
                    # Default validation
                    result = DataQualityResult(
                        rule_id=rule.rule_id,
                        status="warning",
                        score=0.5,
                        total_records=len(data),
                        failed_records=0,
                        passed_records=len(data),
                        details={"message": f"Unknown rule type: {rule.rule_type}"},
                    )

                results.append(result)
                self.results.append(result)

            return results

        def generate_data_profile(
            self, data: list[dict[str, Any]], dataset_name: str
        ) -> dict[str, Any]:
            """Generate comprehensive data profile."""
            if not data:
                return {"error": "No data provided for profiling"}

            total_records = len(data)

            # Field analysis
            field_profiles = {}
            all_fields = set()
            for record in data:
                all_fields.update(record.keys())

            for field in all_fields:
                field_values = [record.get(field) for record in data if field in record]
                non_null_values = [v for v in field_values if v is not None]

                field_profile = {
                    "total_count": len(field_values),
                    "non_null_count": len(non_null_values),
                    "null_count": len(field_values) - len(non_null_values),
                    "null_rate": (len(field_values) - len(non_null_values))
                    / max(len(field_values), 1),
                    "unique_count": len(set(non_null_values)),
                    "uniqueness_rate": len(set(non_null_values))
                    / max(len(non_null_values), 1),
                }

                # Type analysis
                types = [type(v).__name__ for v in non_null_values]
                field_profile["data_types"] = {t: types.count(t) for t in set(types)}
                field_profile["dominant_type"] = (
                    max(
                        field_profile["data_types"], key=field_profile["data_types"].get
                    )
                    if types
                    else None
                )

                # Statistical analysis for numeric fields
                numeric_values = [
                    v for v in non_null_values if isinstance(v, int | float)
                ]
                if numeric_values:
                    field_profile["min_value"] = min(numeric_values)
                    field_profile["max_value"] = max(numeric_values)
                    field_profile["mean_value"] = statistics.mean(numeric_values)
                    field_profile["median_value"] = statistics.median(numeric_values)
                    if len(numeric_values) > 1:
                        field_profile["std_dev"] = statistics.stdev(numeric_values)

                # String analysis
                string_values = [v for v in non_null_values if isinstance(v, str)]
                if string_values:
                    lengths = [len(s) for s in string_values]
                    field_profile["min_length"] = min(lengths)
                    field_profile["max_length"] = max(lengths)
                    field_profile["avg_length"] = statistics.mean(lengths)

                field_profiles[field] = field_profile

            # Overall dataset profile
            profile = {
                "dataset_name": dataset_name,
                "total_records": total_records,
                "total_fields": len(all_fields),
                "profiling_timestamp": datetime.now(tz=UTC).isoformat(),
                "field_profiles": field_profiles,
                "schema_consistency": {
                    "all_records_same_fields": all(
                        set(record.keys()) == all_fields for record in data
                    ),
                    "common_fields": list(all_fields),
                    "variable_fields": [],  # Fields that appear in some but not all records
                },
            }

            # Identify variable fields
            for field in all_fields:
                field_presence = sum(1 for record in data if field in record)
                if field_presence < total_records:
                    profile["schema_consistency"]["variable_fields"].append(
                        {
                            "field": field,
                            "presence_rate": field_presence / total_records,
                        }
                    )

            self.data_profiles[dataset_name] = profile
            return profile

    return DataQualityValidator()


@pytest.fixture
def data_integrity_checker():
    """Data integrity checking utilities."""

    class DataIntegrityChecker:
        def __init__(self):
            self.checks = {}
            self.results = []

        def add_check(self, check: DataIntegrityCheck):
            """Add an integrity check."""
            self.checks[check.check_id] = check

        def check_referential_integrity(
            self,
            data: list[dict[str, Any]],
            reference_data: list[dict[str, Any]],
            check: DataIntegrityCheck,
        ) -> dict[str, Any]:
            """Check referential integrity between datasets."""
            foreign_key = check.constraint.get("foreign_key")
            reference_key = check.constraint.get("reference_key", foreign_key)

            if not foreign_key:
                return {"error": "Foreign key not specified in constraint"}

            # Get reference values
            reference_values = {
                record.get(reference_key)
                for record in reference_data
                if reference_key in record
            }

            # Check for violations
            violations = []
            for record in data:
                if foreign_key in record:
                    foreign_value = record[foreign_key]
                    if (
                        foreign_value not in reference_values
                        and foreign_value is not None
                    ):
                        violations.append(
                            {
                                "record": record,
                                "violation": f"Foreign key {foreign_value} not found in reference data",
                            }
                        )

            return {
                "check_id": check.check_id,
                "check_type": "referential_integrity",
                "total_records": len(data),
                "violations": len(violations),
                "violation_rate": len(violations) / max(len(data), 1),
                "details": violations[:10],  # First 10 violations
                "status": "pass" if len(violations) == 0 else "fail",
            }

        def check_domain_integrity(
            self, data: list[dict[str, Any]], check: DataIntegrityCheck
        ) -> dict[str, Any]:
            """Check domain integrity (valid values)."""
            field_name = check.column_name
            allowed_values = check.constraint.get("allowed_values", [])

            violations = []
            for record in data:
                if field_name in record:
                    value = record[field_name]
                    if value not in allowed_values and value is not None:
                        violations.append(
                            {
                                "record": record,
                                "violation": f"Value '{value}' not in allowed domain: {allowed_values}",
                            }
                        )

            return {
                "check_id": check.check_id,
                "check_type": "domain_integrity",
                "field_name": field_name,
                "total_records": len(data),
                "violations": len(violations),
                "violation_rate": len(violations) / max(len(data), 1),
                "details": violations[:10],
                "status": "pass" if len(violations) == 0 else "fail",
            }

        def check_format_integrity(
            self, data: list[dict[str, Any]], check: DataIntegrityCheck
        ) -> dict[str, Any]:
            """Check format integrity (regex patterns)."""
            field_name = check.column_name
            pattern = check.constraint.get("pattern")

            if not pattern:
                return {"error": "Pattern not specified in constraint"}

            import re

            violations = []

            for record in data:
                if field_name in record:
                    value = record[field_name]
                    if isinstance(value, str) and not re.match(pattern, value):
                        violations.append(
                            {
                                "record": record,
                                "violation": f"Value '{value}' does not match pattern {pattern}",
                            }
                        )

            return {
                "check_id": check.check_id,
                "check_type": "format_integrity",
                "field_name": field_name,
                "pattern": pattern,
                "total_records": len(data),
                "violations": len(violations),
                "violation_rate": len(violations) / max(len(data), 1),
                "details": violations[:10],
                "status": "pass" if len(violations) == 0 else "fail",
            }

        def run_all_checks(
            self,
            data: list[dict[str, Any]],
            reference_datasets: dict[str, list[dict[str, Any]]] | None = None,
        ) -> list[dict[str, Any]]:
            """Run all registered integrity checks."""
            results = []
            reference_datasets = reference_datasets or {}

            for check in self.checks.values():
                if check.check_type == "referential":
                    reference_table = check.constraint.get("reference_table")
                    reference_data = reference_datasets.get(reference_table, [])
                    result = self.check_referential_integrity(
                        data, reference_data, check
                    )

                elif check.check_type == "domain":
                    result = self.check_domain_integrity(data, check)

                elif check.check_type == "format":
                    result = self.check_format_integrity(data, check)

                else:
                    result = {
                        "check_id": check.check_id,
                        "error": f"Unknown check type: {check.check_type}",
                    }

                results.append(result)
                self.results.append(result)

            return results

    return DataIntegrityChecker()


@pytest.fixture
def mock_data_generator():
    """Generate mock data for testing."""

    class MockDataGenerator:
        def __init__(self):
            self.seed = 42
            random.seed(self.seed)

        def generate_document_records(
            self, count: int = 100, introduce_quality_issues: bool = False
        ) -> list[dict[str, Any]]:
            """Generate mock document records."""
            records = []

            for i in range(count):
                record = {
                    "id": f"doc_{i:04d}",
                    "url": f"https://example.com/doc/{i}",
                    "title": f"Document {i}",
                    "content": f"This is the content of document {i}. "
                    * random.randint(5, 20),
                    "created_at": datetime.now(tz=UTC)
                    - timedelta(days=random.randint(0, 365)),
                    "updated_at": datetime.now(tz=UTC)
                    - timedelta(days=random.randint(0, 30)),
                    "collection": random.choice(
                        ["tech", "science", "business", "general"]
                    ),
                    "status": random.choice(["active", "archived", "pending"]),
                    "word_count": random.randint(100, 2000),
                    "embedding_model": "text-embedding-ada-002",
                    "metadata": {
                        "source": random.choice(["web", "api", "upload"]),
                        "language": "en",
                        "tags": random.sample(
                            ["ai", "ml", "data", "tech", "science"],
                            random.randint(1, 3),
                        ),
                    },
                }

                # Introduce quality issues if requested
                if introduce_quality_issues:
                    issue_type = random.choice(
                        ["missing_field", "invalid_format", "duplicate", "inconsistent"]
                    )

                    if issue_type == "missing_field" and random.random() < 0.1:
                        # Remove a required field
                        field_to_remove = random.choice(["title", "url", "content"])
                        record.pop(field_to_remove, None)

                    elif issue_type == "invalid_format" and random.random() < 0.05:
                        # Invalid URL format
                        record["url"] = f"invalid-url-{i}"

                    elif issue_type == "duplicate" and random.random() < 0.02:
                        # Duplicate ID
                        record["id"] = records[-1]["id"] if records else "doc_0000"

                    elif issue_type == "inconsistent" and random.random() < 0.05:
                        # Inconsistent dates (updated before created)
                        record["updated_at"] = record["created_at"] - timedelta(days=1)

                records.append(record)

            return records

        def generate_user_records(self, count: int = 50) -> list[dict[str, Any]]:
            """Generate mock user records."""
            records = []

            for i in range(count):
                record = {
                    "user_id": f"user_{i:04d}",
                    "email": f"user{i}@example.com",
                    "name": f"User {i}",
                    "role": random.choice(["admin", "user", "viewer"]),
                    "created_at": datetime.now(tz=UTC)
                    - timedelta(days=random.randint(30, 730)),
                    "last_login": datetime.now(tz=UTC)
                    - timedelta(days=random.randint(0, 30)),
                    "active": random.choice([True, False]),
                    "preferences": {
                        "theme": random.choice(["light", "dark"]),
                        "notifications": random.choice([True, False]),
                    },
                }
                records.append(record)

            return records

    return MockDataGenerator()


@pytest.fixture
def data_quality_test_data():
    """Provide test data for data quality testing."""
    return {
        "sample_rules": [
            DataQualityRule(
                rule_id="completeness_id",
                name="ID Completeness",
                description="All records must have an ID",
                rule_type="completeness",
                severity="critical",
                field_name="id",
                threshold=1.0,
            ),
            DataQualityRule(
                rule_id="validity_url",
                name="URL Format Validity",
                description="URLs must follow valid format",
                rule_type="validity",
                severity="high",
                field_name="url",
                pattern=r"^https?://[^\s/$.?#].[^\s]*$",
                threshold=0.95,
            ),
            DataQualityRule(
                rule_id="uniqueness_id",
                name="ID Uniqueness",
                description="IDs must be unique",
                rule_type="uniqueness",
                severity="critical",
                field_name="id",
                threshold=1.0,
            ),
            DataQualityRule(
                rule_id="consistency_dates",
                name="Date Consistency",
                description="Updated date should be after created date",
                rule_type="consistency",
                severity="medium",
                threshold=0.90,
                reference_data={
                    "consistency_rules": [
                        {
                            "type": "field_relationship",
                            "field1": "updated_at",
                            "field2": "created_at",
                            "relationship": "greater",
                        }
                    ]
                },
            ),
        ],
        "sample_integrity_checks": [
            DataIntegrityCheck(
                check_id="ref_user_collection",
                name="User Collection Reference",
                check_type="referential",
                table_name="documents",
                column_name="created_by",
                constraint={
                    "foreign_key": "created_by",
                    "reference_table": "users",
                    "reference_key": "user_id",
                },
                severity="high",
            ),
            DataIntegrityCheck(
                check_id="domain_status",
                name="Status Domain Check",
                check_type="domain",
                table_name="documents",
                column_name="status",
                constraint={
                    "allowed_values": ["active", "archived", "pending", "deleted"],
                },
                severity="medium",
            ),
            DataIntegrityCheck(
                check_id="format_email",
                name="Email Format Check",
                check_type="format",
                table_name="users",
                column_name="email",
                constraint={
                    "pattern": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                },
                severity="high",
            ),
        ],
    }


# Pytest markers for data quality test categorization
def pytest_configure(config):
    """Configure data quality testing markers."""
    config.addinivalue_line("markers", "data_quality: mark test as data quality test")
    config.addinivalue_line(
        "markers", "data_validation: mark test as data validation test"
    )
    config.addinivalue_line(
        "markers", "data_consistency: mark test as data consistency test"
    )
    config.addinivalue_line(
        "markers", "data_integrity: mark test as data integrity test"
    )
    config.addinivalue_line(
        "markers", "data_transformation: mark test as data transformation test"
    )
    config.addinivalue_line(
        "markers", "data_migration: mark test as data migration test"
    )
    config.addinivalue_line(
        "markers", "data_profiling: mark test as data profiling test"
    )
