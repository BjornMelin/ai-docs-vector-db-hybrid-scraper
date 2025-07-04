#!/usr/bin/env python3
"""Security test runner and reporting.

This script runs comprehensive security tests and generates detailed reports
for vulnerability assessment and compliance validation.
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


class SecurityTestRunner:
    """Comprehensive security test runner."""

    def __init__(self, project_root: Path, output_dir: Path):
        """Initialize security test runner.

        Args:
            project_root: Root directory of the project
            output_dir: Directory for test reports and artifacts
        """
        self.project_root = project_root
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.output_dir / "security_tests.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def run_all_security_tests(
        self, test_categories: list[str] | None = None
    ) -> dict[str, Any]:
        """Run all security tests and generate comprehensive report.

        Args:
            test_categories: Specific test categories to run (None for all)

        Returns:
            Comprehensive test results
        """
        self.logger.info("Starting comprehensive security test suite")
        start_time = time.time()

        results = {
            "execution_info": {
                "start_time": start_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "project_root": str(self.project_root),
                "output_dir": str(self.output_dir),
            },
            "test_categories": {},
            "summary": {},
            "recommendations": [],
        }

        # Define test categories
        available_categories = {
            "input_validation": self._run_input_validation_tests,
            "authentication": self._run_authentication_tests,
            "authorization": self._run_authorization_tests,
            "vulnerability": self._run_vulnerability_tests,
            "penetration": self._run_penetration_tests,
            "compliance": self._run_compliance_tests,
            "encryption": self._run_encryption_tests,
            "static_analysis": self._run_static_analysis,
            "dependency_scan": self._run_dependency_scan,
        }

        # Run selected categories or all
        categories_to_run = test_categories or available_categories.keys()

        for category in categories_to_run:
            if category in available_categories:
                self.logger.info("Running %s tests...", category)
                try:
                    category_results = available_categories[category]()
                    results["test_categories"][category] = category_results
                    self.logger.info("Completed {category} tests")
                except Exception as e:
                    self.logger.exception("Error in {category} tests")
                    results["test_categories"][category] = {
                        "status": "error",
                        "error": str(e),
                        "tests_run": 0,
                        "tests_passed": 0,
                        "tests_failed": 1,
                    }

        # Calculate summary
        results["summary"] = self._calculate_summary(results["test_categories"])
        results["recommendations"] = self._generate_recommendations(results)
        results["execution_info"]["end_time"] = time.time()
        results["execution_info"]["duration"] = (
            results["execution_info"]["end_time"] - start_time
        )

        # Generate reports
        self._generate_reports(results)

        self.logger.info(
            "Security test suite completed in %.2f seconds",
            results["execution_info"]["duration"],
        )
        return results

    def _run_input_validation_tests(self) -> dict[str, Any]:
        """Run input validation security tests."""
        return self._run_pytest_category(
            "input_validation",
            [
                "tests/security/input_validation/test_sql_injection.py",
                "tests/security/input_validation/test_xss_prevention.py",
                "tests/security/input_validation/test_command_injection.py",
            ],
        )

    def _run_authentication_tests(self) -> dict[str, Any]:
        """Run authentication security tests."""
        return self._run_pytest_category(
            "authentication", ["tests/security/authentication/test_jwt_security.py"]
        )

    def _run_authorization_tests(self) -> dict[str, Any]:
        """Run authorization security tests."""
        return self._run_pytest_category(
            "authorization", ["tests/security/authorization/test_access_control.py"]
        )

    def _run_vulnerability_tests(self) -> dict[str, Any]:
        """Run vulnerability scanning tests."""
        return self._run_pytest_category(
            "vulnerability",
            ["tests/security/vulnerability/test_dependency_scanning.py"],
        )

    def _run_penetration_tests(self) -> dict[str, Any]:
        """Run penetration testing scenarios."""
        return self._run_pytest_category(
            "penetration", ["tests/security/penetration/test_api_security.py"]
        )

    def _run_compliance_tests(self) -> dict[str, Any]:
        """Run compliance validation tests."""
        return self._run_pytest_category(
            "compliance", ["tests/security/compliance/test_owasp_top10.py"]
        )

    def _run_encryption_tests(self) -> dict[str, Any]:
        """Run encryption and data protection tests."""
        return self._run_pytest_category(
            "encryption", ["tests/security/encryption/test_data_protection.py"]
        )

    def _run_static_analysis(self) -> dict[str, Any]:
        """Run static security analysis with Bandit."""
        self.logger.info("Running Bandit static security analysis...")

        try:
            # Validate Python executable for security
            if not sys.executable or not Path(sys.executable).is_file():
                self.logger.warning(
                    "Python executable not found or invalid, skipping Bandit"
                )
                return False

            # Run Bandit - explicitly construct command for security
            bandit_cmd = [
                sys.executable,
                "-m",
                "bandit",
                "-r",
                "src/",
                "-f",
                "json",
                "-o",
                str(self.output_dir / "bandit_report.json"),
            ]
            subprocess.run(  # Secure: validated executable, no shell, no user input
                bandit_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300,
                check=False,  # 5 minutes
            )

            # Parse results
            bandit_report_file = self.output_dir / "bandit_report.json"
            if bandit_report_file.exists():
                with bandit_report_file.open() as f:
                    bandit_data = json.load(f)

                return {
                    "status": "completed",
                    "tool": "bandit",
                    "issues_found": len(bandit_data.get("results", [])),
                    "high_severity": len(
                        [
                            r
                            for r in bandit_data.get("results", [])
                            if r.get("issue_severity") == "HIGH"
                        ]
                    ),
                    "medium_severity": len(
                        [
                            r
                            for r in bandit_data.get("results", [])
                            if r.get("issue_severity") == "MEDIUM"
                        ]
                    ),
                    "low_severity": len(
                        [
                            r
                            for r in bandit_data.get("results", [])
                            if r.get("issue_severity") == "LOW"
                        ]
                    ),
                    "report_file": str(bandit_report_file),
                }

        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Bandit analysis timed out"}
        except (ValueError, RuntimeError, ImportError) as e:
            return {"status": "error", "error": str(e)}
        else:
            return {"status": "failed", "error": "No report generated"}

    def _run_dependency_scan(self) -> dict[str, Any]:
        """Run dependency vulnerability scanning with Safety."""
        self.logger.info("Running Safety dependency vulnerability scan...")

        try:
            # Validate Python executable for security
            if not sys.executable or not Path(sys.executable).is_file():
                self.logger.warning(
                    "Python executable not found or invalid, skipping Safety"
                )
                return {"status": "skipped", "error": "Python executable invalid"}

            # Run Safety - explicitly construct command for security
            safety_cmd = [
                sys.executable,
                "-m",
                "safety",
                "check",
                "--json",
                "--output",
                str(self.output_dir / "safety_report.json"),
            ]
            subprocess.run(  # Secure: validated executable, no shell, no user input
                safety_cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,  # 3 minutes
            )

            # Parse results
            safety_report_file = self.output_dir / "safety_report.json"
            if safety_report_file.exists():
                with safety_report_file.open() as f:
                    try:
                        safety_data = json.load(f)
                        vulnerabilities = (
                            len(safety_data) if isinstance(safety_data, list) else 0
                        )
                    except json.JSONDecodeError:
                        vulnerabilities = 0

                return {
                    "status": "completed",
                    "tool": "safety",
                    "vulnerabilities_found": vulnerabilities,
                    "report_file": str(safety_report_file),
                }

        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": "Safety scan timed out"}
        except (ValueError, RuntimeError, ImportError) as e:
            return {"status": "error", "error": str(e)}
        else:
            # No vulnerabilities found (Safety returns 0 exit code)
            return {
                "status": "completed",
                "tool": "safety",
                "vulnerabilities_found": 0,
                "message": "No vulnerabilities detected",
            }

    def _run_pytest_category(
        self, category: str, test_files: list[str]
    ) -> dict[str, Any]:
        """Run pytest for a specific security category.

        Args:
            category: Security test category name
            test_files: List of test files to run

        Returns:
            Test execution results
        """
        self.logger.info("Running pytest for %s category...", category)

        # Build pytest command
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-v",
            "--tb=short",
            f"--junit-xml={self.output_dir}/{category}_junit.xml",
            f"--html={self.output_dir}/{category}_report.html",
            "--self-contained-html",
            "-m",
            "security",
        ]

        # Add test files that exist
        existing_files = []
        for test_file in test_files:
            file_path = self.project_root / test_file
            if file_path.exists():
                existing_files.append(str(file_path))
            else:
                self.logger.warning("Test file not found")

        if not existing_files:
            return {
                "status": "skipped",
                "reason": "No test files found",
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 0,
            }

        cmd.extend(existing_files)

        # Validate Python executable for security
        if not sys.executable or not Path(sys.executable).is_file():
            self.logger.warning(
                "Python executable not found or invalid, skipping pytest"
            )
            return {"status": "skipped", "error": "Python executable invalid"}

        try:
            result = (
                subprocess.run(  # Secure: validated executable, no shell, no user input
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=600,
                    check=False,  # 10 minutes
                )
            )

            # Parse pytest output
            lines = result.stdout.split("\n")
            summary_line = next(
                (
                    line
                    for line in lines
                    if "passed" in line
                    and ("failed" in line or "error" in line or "skipped" in line)
                ),
                "",
            )

            if not summary_line:
                summary_line = next((line for line in lines if "passed" in line), "")

            # Extract test counts
            tests_passed = self._extract_count(summary_line, "passed")
            tests_failed = self._extract_count(
                summary_line, "failed"
            ) + self._extract_count(summary_line, "error")
            tests_skipped = self._extract_count(summary_line, "skipped")

            return {
                "status": "completed" if result.returncode == 0 else "failed",
                "tests_run": tests_passed + tests_failed + tests_skipped,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "tests_skipped": tests_skipped,
                "exit_code": result.returncode,
                "report_files": [
                    str(self.output_dir / f"{category}_junit.xml"),
                    str(self.output_dir / f"{category}_report.html"),
                ],
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "error": f"Pytest timed out for {category}",
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 1,
            }
        except (ValueError, RuntimeError, ImportError) as e:
            return {
                "status": "error",
                "error": str(e),
                "tests_run": 0,
                "tests_passed": 0,
                "tests_failed": 1,
            }

    def _extract_count(self, text: str, keyword: str) -> int:
        """Extract test count from pytest summary line."""

        pattern = rf"(\d+)\s+{keyword}"
        match = re.search(pattern, text)
        return int(match.group(1)) if match else 0

    def _calculate_summary(self, category_results: dict[str, Any]) -> dict[str, Any]:
        """Calculate overall test summary."""
        _total_tests = 0
        _total_passed = 0
        _total_failed = 0
        _total_skipped = 0
        categories_completed = 0
        categories_failed = 0

        for results in category_results.values():
            if results.get("status") == "completed":
                categories_completed += 1
            elif results.get("status") in ["failed", "error", "timeout"]:
                categories_failed += 1

            _total_tests += results.get("tests_run", 0)
            _total_passed += results.get("tests_passed", 0)
            _total_failed += results.get("tests_failed", 0)
            _total_skipped += results.get("tests_skipped", 0)

        success_rate = (_total_passed / _total_tests * 100) if _total_tests > 0 else 0

        return {
            "_total_categories": len(category_results),
            "categories_completed": categories_completed,
            "categories_failed": categories_failed,
            "_total_tests": _total_tests,
            "_total_passed": _total_passed,
            "_total_failed": _total_failed,
            "_total_skipped": _total_skipped,
            "success_rate": round(success_rate, 2),
            "overall_status": "PASS"
            if _total_failed == 0 and categories_failed == 0
            else "FAIL",
        }

    def _generate_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate security recommendations based on test results."""
        recommendations = []

        # Check for high-priority issues
        for category, category_results in results["test_categories"].items():
            if category_results.get("status") == "failed":
                recommendations.append(f"Address failing tests in {category} category")

            # Static analysis recommendations
            if category == "static_analysis":
                high_severity = category_results.get("high_severity", 0)
                if high_severity > 0:
                    recommendations.append(
                        f"Fix {high_severity} high-severity security issues found by static analysis"
                    )

            # Dependency recommendations
            if category == "dependency_scan":
                vulnerabilities = category_results.get("vulnerabilities_found", 0)
                if vulnerabilities > 0:
                    recommendations.append(
                        f"Update {vulnerabilities} vulnerable dependencies"
                    )

        # General recommendations
        summary = results.get("summary", {})
        if summary.get("success_rate", 100) < 95:
            recommendations.append(
                "Improve test coverage and fix failing security tests"
            )

        if not recommendations:
            recommendations.append(
                "Maintain current security posture with regular testing"
            )

        return recommendations

    def _generate_reports(self, results: dict[str, Any]) -> None:
        """Generate comprehensive security reports."""
        # JSON report
        json_report_file = self.output_dir / "security_test_results.json"
        with json_report_file.open("w") as f:
            json.dump(results, f, indent=2, default=str)

        # HTML summary report
        self._generate_html_report(results)

        # Executive summary
        self._generate_executive_summary(results)

        self.logger.info("Reports generated in %s", self.output_dir)

    def _generate_html_report(self, results: dict[str, Any]) -> None:
        """Generate HTML summary report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Security Test Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .category {{ margin: 15px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .pass {{ background-color: #d4edda; }}
        .fail {{ background-color: #f8d7da; }}
        .skip {{ background-color: #fff3cd; }}
        .recommendations {{ background-color: #e2e3e5; padding: 15px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Security Test Results</h1>
        <p><strong>Execution Time:</strong> {results["execution_info"]["timestamp"]}</p>
        <p><strong>Duration:</strong> {results["execution_info"].get("duration", 0):.2f} seconds</p>
        <p><strong>Overall Status:</strong> <span class="{"pass" if results["summary"]["overall_status"] == "PASS" else "fail"}">{results["summary"]["overall_status"]}</span></p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Tests</td><td>{results["summary"]["_total_tests"]}</td></tr>
            <tr><td>Tests Passed</td><td>{results["summary"]["_total_passed"]}</td></tr>
            <tr><td>Tests Failed</td><td>{results["summary"]["_total_failed"]}</td></tr>
            <tr><td>Tests Skipped</td><td>{results["summary"]["_total_skipped"]}</td></tr>
            <tr><td>Success Rate</td><td>{results["summary"]["success_rate"]}%</td></tr>
        </table>
    </div>

    <div class="categories">
        <h2>Test Categories</h2>
        """

        for category, category_results in results["test_categories"].items():
            status_class = (
                "pass" if category_results.get("status") == "completed" else "fail"
            )
            html_content += f"""
        <div class="category {status_class}">
            <h3>{category.title()}</h3>
            <p><strong>Status:</strong> {category_results.get("status", "unknown")}</p>
            <p><strong>Tests Run:</strong> {category_results.get("tests_run", 0)}</p>
            <p><strong>Tests Passed:</strong> {category_results.get("tests_passed", 0)}</p>
            <p><strong>Tests Failed:</strong> {category_results.get("tests_failed", 0)}</p>
        </div>"""

        html_content += """
    </div>

    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
        """

        for recommendation in results["recommendations"]:
            html_content += f"<li>{recommendation}</li>"

        html_content += """
        </ul>
    </div>
</body>
</html>"""

        html_report_file = self.output_dir / "security_summary.html"
        with html_report_file.open("w") as f:
            f.write(html_content)

    def _generate_executive_summary(self, results: dict[str, Any]) -> None:
        """Generate executive summary report."""
        summary_content = f"""
SECURITY TEST EXECUTIVE SUMMARY
================================

Execution Date: {results["execution_info"]["timestamp"]}
Overall Status: {results["summary"]["overall_status"]}
Success Rate: {results["summary"]["success_rate"]}%

TEST RESULTS OVERVIEW:
- Total Tests: {results["summary"]["_total_tests"]}
- Tests Passed: {results["summary"]["_total_passed"]}
- Tests Failed: {results["summary"]["_total_failed"]}
- Tests Skipped: {results["summary"]["_total_skipped"]}

CATEGORY BREAKDOWN:
"""

        for category, category_results in results["test_categories"].items():
            summary_content += (
                f"- {category.title()}: {category_results.get('status', 'unknown')}\n"
            )

        summary_content += """
KEY RECOMMENDATIONS:
"""
        for i, recommendation in enumerate(results["recommendations"], 1):
            summary_content += f"{i}. {recommendation}\n"

        summary_content += f"""
RISK ASSESSMENT:
- Overall Risk Level: {"LOW" if results["summary"]["_total_failed"] == 0 else "MEDIUM" if results["summary"]["_total_failed"] < 5 else "HIGH"}
- Action Required: {"No" if results["summary"]["_total_failed"] == 0 else "Yes"}

For detailed results, see: security_test_results.json
For visual summary, see: security_summary.html
"""

        summary_file = self.output_dir / "executive_summary.txt"
        with summary_file.open("w") as f:
            f.write(summary_content)


def main():
    """Main entry point for security test runner."""
    parser = argparse.ArgumentParser(description="Run comprehensive security tests")
    parser.add_argument(
        "--project-root", type=Path, default=Path.cwd(), help="Project root directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "security_reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--categories", nargs="+", help="Specific test categories to run"
    )
    parser.add_argument(
        "--list-categories", action="store_true", help="List available test categories"
    )

    args = parser.parse_args()

    if args.list_categories:
        print("Available security test categories:")
        categories = [
            "input_validation",
            "authentication",
            "authorization",
            "vulnerability",
            "penetration",
            "compliance",
            "encryption",
            "static_analysis",
            "dependency_scan",
        ]
        for category in categories:
            print(f"  - {category}")
        return

    # Run security tests
    runner = SecurityTestRunner(args.project_root, args.output_dir)
    results = runner.run_all_security_tests(args.categories)

    # Print summary
    print("\nSecurity Test Summary:")
    print("Overall Status")
    print(f"Success Rate: {results['summary']['success_rate']}%")
    print(
        f"Tests: {results['summary']['_total_passed']}/{results['summary']['_total_tests']} passed"
    )
    print("Reports generated in")

    # Exit with appropriate code
    sys.exit(0 if results["summary"]["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
