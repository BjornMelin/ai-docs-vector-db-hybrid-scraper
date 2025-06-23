import typing
"""Contract test runner and reporting utilities.

This module provides utilities for running contract tests, generating reports,
and managing contract validation results.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import pytest


class ContractTestResult(Enum):
    """Contract test result types."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class ContractValidationResult:
    """Result of a contract validation."""

    test_name: str
    contract_type: str
    result: ContractTestResult
    details: dict[str, Any]
    errors: list[str]
    warnings: list[str]
    execution_time_ms: float
    timestamp: datetime


@dataclass
class ContractTestReport:
    """Complete contract test report."""

    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    warning_tests: int
    execution_time_ms: float
    test_results: list[ContractValidationResult]
    summary: dict[str, Any]
    generated_at: datetime


class ContractTestRunner:
    """Contract test runner with reporting capabilities."""

    def __init__(self):
        """Initialize contract test runner."""
        self.results: list[ContractValidationResult] = []
        self.start_time: typing.Optional[datetime] = None
        self.end_time: typing.Optional[datetime] = None

    def start_test_session(self):
        """Start a contract testing session."""
        self.start_time = datetime.now()
        self.results = []

    def end_test_session(self):
        """End a contract testing session."""
        self.end_time = datetime.now()

    def add_result(self, result: ContractValidationResult):
        """Add a test result to the session."""
        self.results.append(result)

    def generate_report(self) -> ContractTestReport:
        """Generate a comprehensive test report."""
        if not self.start_time or not self.end_time:
            raise ValueError("Test session not properly started/ended")

        total_time = (self.end_time - self.start_time).total_seconds() * 1000

        # Count results by type
        passed = sum(1 for r in self.results if r.result == ContractTestResult.PASSED)
        failed = sum(1 for r in self.results if r.result == ContractTestResult.FAILED)
        skipped = sum(1 for r in self.results if r.result == ContractTestResult.SKIPPED)
        warning = sum(1 for r in self.results if r.result == ContractTestResult.WARNING)

        # Generate summary
        summary = {
            "success_rate": passed / len(self.results) if self.results else 0,
            "contract_types": list({r.contract_type for r in self.results}),
            "avg_execution_time_ms": sum(r.execution_time_ms for r in self.results)
            / len(self.results)
            if self.results
            else 0,
            "total_errors": sum(len(r.errors) for r in self.results),
            "total_warnings": sum(len(r.warnings) for r in self.results),
        }

        return ContractTestReport(
            total_tests=len(self.results),
            passed_tests=passed,
            failed_tests=failed,
            skipped_tests=skipped,
            warning_tests=warning,
            execution_time_ms=total_time,
            test_results=self.results,
            summary=summary,
            generated_at=self.end_time,
        )

    def save_report(self, report: ContractTestReport, output_path: Path):
        """Save test report to file."""
        report_data = {
            "metadata": {
                "total_tests": report.total_tests,
                "passed_tests": report.passed_tests,
                "failed_tests": report.failed_tests,
                "skipped_tests": report.skipped_tests,
                "warning_tests": report.warning_tests,
                "execution_time_ms": report.execution_time_ms,
                "generated_at": report.generated_at.isoformat(),
            },
            "summary": report.summary,
            "test_results": [
                {
                    "test_name": result.test_name,
                    "contract_type": result.contract_type,
                    "result": result.result.value,
                    "details": result.details,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "execution_time_ms": result.execution_time_ms,
                    "timestamp": result.timestamp.isoformat(),
                }
                for result in report.test_results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

    def load_report(self, input_path: Path) -> ContractTestReport:
        """Load test report from file."""
        with open(input_path) as f:
            data = json.load(f)

        metadata = data["metadata"]
        test_results = [
            ContractValidationResult(
                test_name=result["test_name"],
                contract_type=result["contract_type"],
                result=ContractTestResult(result["result"]),
                details=result["details"],
                errors=result["errors"],
                warnings=result["warnings"],
                execution_time_ms=result["execution_time_ms"],
                timestamp=datetime.fromisoformat(result["timestamp"]),
            )
            for result in data["test_results"]
        ]

        return ContractTestReport(
            total_tests=metadata["total_tests"],
            passed_tests=metadata["passed_tests"],
            failed_tests=metadata["failed_tests"],
            skipped_tests=metadata["skipped_tests"],
            warning_tests=metadata["warning_tests"],
            execution_time_ms=metadata["execution_time_ms"],
            test_results=test_results,
            summary=data["summary"],
            generated_at=datetime.fromisoformat(metadata["generated_at"]),
        )


class TestContractRunner:
    """Test the contract test runner functionality."""

    @pytest.mark.contract
    def test_contract_test_runner_basic_functionality(self):
        """Test basic contract test runner functionality."""
        runner = ContractTestRunner()

        # Start session
        runner.start_test_session()
        assert runner.start_time is not None
        assert len(runner.results) == 0

        # Add test results
        result1 = ContractValidationResult(
            test_name="test_search_contract",
            contract_type="api_contract",
            result=ContractTestResult.PASSED,
            details={"endpoint": "/api/search", "method": "GET"},
            errors=[],
            warnings=[],
            execution_time_ms=45.0,
            timestamp=datetime.now(),
        )

        result2 = ContractValidationResult(
            test_name="test_schema_validation",
            contract_type="json_schema",
            result=ContractTestResult.FAILED,
            details={"schema": "search_result"},
            errors=["Missing required field: id"],
            warnings=[],
            execution_time_ms=12.0,
            timestamp=datetime.now(),
        )

        runner.add_result(result1)
        runner.add_result(result2)

        assert len(runner.results) == 2

        # End session
        runner.end_test_session()
        assert runner.end_time is not None

        # Generate report
        report = runner.generate_report()

        assert report.total_tests == 2
        assert report.passed_tests == 1
        assert report.failed_tests == 1
        assert report.skipped_tests == 0
        assert report.warning_tests == 0
        assert report.execution_time_ms > 0
        assert len(report.test_results) == 2

        # Verify summary
        summary = report.summary
        assert summary["success_rate"] == 0.5
        assert "api_contract" in summary["contract_types"]
        assert "json_schema" in summary["contract_types"]
        assert summary["total_errors"] == 1
        assert summary["total_warnings"] == 0

    @pytest.mark.contract
    def test_contract_report_serialization(self, tmp_path):
        """Test contract report serialization and deserialization."""
        runner = ContractTestRunner()
        runner.start_test_session()

        # Add sample results
        result = ContractValidationResult(
            test_name="serialization_test",
            contract_type="openapi",
            result=ContractTestResult.PASSED,
            details={"spec_version": "3.0.3"},
            errors=[],
            warnings=["Deprecated field used"],
            execution_time_ms=25.0,
            timestamp=datetime.now(),
        )

        runner.add_result(result)
        runner.end_test_session()

        # Generate and save report
        report = runner.generate_report()
        output_path = tmp_path / "test_report.json"
        runner.save_report(report, output_path)

        # Verify file exists and has content
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Load and verify report
        loaded_report = runner.load_report(output_path)

        assert loaded_report.total_tests == report.total_tests
        assert loaded_report.passed_tests == report.passed_tests
        assert loaded_report.summary == report.summary
        assert len(loaded_report.test_results) == len(report.test_results)

        # Verify individual result
        loaded_result = loaded_report.test_results[0]
        assert loaded_result.test_name == result.test_name
        assert loaded_result.contract_type == result.contract_type
        assert loaded_result.result == result.result
        assert loaded_result.details == result.details
        assert loaded_result.warnings == result.warnings

    @pytest.mark.contract
    def test_contract_runner_with_multiple_contract_types(self):
        """Test contract runner with multiple contract types."""
        runner = ContractTestRunner()
        runner.start_test_session()

        # Add results for different contract types
        contract_types = [
            "api_contract",
            "json_schema",
            "openapi",
            "pact",
            "consumer_driven",
        ]

        for i, contract_type in enumerate(contract_types):
            result = ContractValidationResult(
                test_name=f"test_{contract_type}_{i}",
                contract_type=contract_type,
                result=ContractTestResult.PASSED
                if i % 2 == 0
                else ContractTestResult.FAILED,
                details={"test_id": i},
                errors=[f"Error in {contract_type}"] if i % 2 == 1 else [],
                warnings=[],
                execution_time_ms=float(10 + i * 5),
                timestamp=datetime.now(),
            )
            runner.add_result(result)

        runner.end_test_session()
        report = runner.generate_report()

        # Verify report statistics
        assert report.total_tests == 5
        assert report.passed_tests == 3  # Even indices
        assert report.failed_tests == 2  # Odd indices
        assert len(report.summary["contract_types"]) == 5
        assert report.summary["total_errors"] == 2

    @pytest.mark.contract
    def test_contract_runner_performance_metrics(self):
        """Test contract runner performance metrics calculation."""
        runner = ContractTestRunner()
        runner.start_test_session()

        # Add results with varying execution times
        execution_times = [10.0, 25.0, 15.0, 50.0, 30.0]

        for i, exec_time in enumerate(execution_times):
            result = ContractValidationResult(
                test_name=f"performance_test_{i}",
                contract_type="performance",
                result=ContractTestResult.PASSED,
                details={},
                errors=[],
                warnings=[],
                execution_time_ms=exec_time,
                timestamp=datetime.now(),
            )
            runner.add_result(result)

        runner.end_test_session()
        report = runner.generate_report()

        # Verify performance metrics
        expected_avg = sum(execution_times) / len(execution_times)
        assert abs(report.summary["avg_execution_time_ms"] - expected_avg) < 0.1
        assert report.execution_time_ms > 0

        # Find slowest and fastest tests
        slowest_test = max(report.test_results, key=lambda r: r.execution_time_ms)
        fastest_test = min(report.test_results, key=lambda r: r.execution_time_ms)

        assert slowest_test.execution_time_ms == 50.0
        assert fastest_test.execution_time_ms == 10.0

    @pytest.mark.contract
    def test_contract_runner_error_aggregation(self):
        """Test contract runner error and warning aggregation."""
        runner = ContractTestRunner()
        runner.start_test_session()

        # Add results with various errors and warnings
        test_cases = [
            {
                "errors": ["Schema validation failed", "Missing required field"],
                "warnings": ["Deprecated field used"],
            },
            {"errors": [], "warnings": ["Performance warning", "Style warning"]},
            {"errors": ["Network timeout"], "warnings": []},
            {"errors": [], "warnings": []},
        ]

        for i, case in enumerate(test_cases):
            result = ContractValidationResult(
                test_name=f"error_test_{i}",
                contract_type="error_test",
                result=ContractTestResult.FAILED
                if case["errors"]
                else ContractTestResult.PASSED,
                details={},
                errors=case["errors"],
                warnings=case["warnings"],
                execution_time_ms=20.0,
                timestamp=datetime.now(),
            )
            runner.add_result(result)

        runner.end_test_session()
        report = runner.generate_report()

        # Verify error and warning aggregation
        assert report.summary["total_errors"] == 3  # 2 + 0 + 1 + 0
        assert report.summary["total_warnings"] == 3  # 1 + 2 + 0 + 0

        # Verify test classification
        assert report.failed_tests == 2  # Tests with errors
        assert report.passed_tests == 2  # Tests without errors

    @pytest.mark.contract
    def test_contract_runner_integration_with_fixtures(
        self, api_contract_validator, json_schema_validator, contract_test_data
    ):
        """Test contract runner integration with actual contract fixtures."""
        runner = ContractTestRunner()
        runner.start_test_session()

        # Test 1: JSON Schema validation
        start_time = datetime.now()

        search_schema = contract_test_data["json_schemas"]["search_result"]
        json_schema_validator.register_schema("search_result", search_schema)

        valid_data = {"id": "doc1", "title": "Test Document", "score": 0.95}

        schema_validation = json_schema_validator.validate_data(
            valid_data, "search_result"
        )

        exec_time = (datetime.now() - start_time).total_seconds() * 1000

        schema_result = ContractValidationResult(
            test_name="json_schema_integration_test",
            contract_type="json_schema",
            result=ContractTestResult.PASSED
            if schema_validation["valid"]
            else ContractTestResult.FAILED,
            details={"schema_name": "search_result"},
            errors=schema_validation.get("errors", []),
            warnings=[],
            execution_time_ms=exec_time,
            timestamp=datetime.now(),
        )

        runner.add_result(schema_result)

        # Test 2: API Contract validation
        start_time = datetime.now()

        api_contract_validator.register_contract(
            "/api/search", contract_test_data["api_contracts"]["/api/search"]
        )

        api_validation = api_contract_validator.validate_request(
            "/api/search", "GET", params={"q": "test query", "limit": 10}
        )

        exec_time = (datetime.now() - start_time).total_seconds() * 1000

        api_result = ContractValidationResult(
            test_name="api_contract_integration_test",
            contract_type="api_contract",
            result=ContractTestResult.PASSED
            if api_validation["valid"]
            else ContractTestResult.FAILED,
            details={"endpoint": "/api/search", "method": "GET"},
            errors=api_validation.get("errors", []),
            warnings=[],
            execution_time_ms=exec_time,
            timestamp=datetime.now(),
        )

        runner.add_result(api_result)

        runner.end_test_session()
        report = runner.generate_report()

        # Verify integration test results
        assert report.total_tests == 2
        assert report.passed_tests == 2
        assert report.failed_tests == 0

        # Verify contract types are captured
        contract_types = report.summary["contract_types"]
        assert "json_schema" in contract_types
        assert "api_contract" in contract_types

        # Verify execution metrics
        assert report.summary["avg_execution_time_ms"] > 0
        assert report.execution_time_ms > 0
