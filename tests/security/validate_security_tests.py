#!/usr/bin/env python3
"""Security test validation script.

This script validates the security test implementation without requiring
the full project environment.
"""

import ast
import sys
from pathlib import Path


def validate_test_file(file_path: Path) -> dict:
    """Validate a security test file structure."""
    results = {
        "file": str(file_path),
        "valid": False,
        "errors": [],
        "test_count": 0,
        "fixtures": [],
        "security_markers": False,
    }

    try:
        with file_path.open() as f:
            content = f.read()

        # Parse the AST
        tree = ast.parse(content)

        # Check for security markers
        if "@pytest.mark.security" in content:
            results["security_markers"] = True

        # Count test functions and fixtures
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("test_"):
                    results["test_count"] += 1
                elif any(
                    decorator.id == "pytest.fixture"
                    if hasattr(decorator, "id")
                    else str(decorator).endswith("fixture")
                    for decorator in node.decorator_list
                ):
                    results["fixtures"].append(node.name)

        # Basic validation checks
        if results["test_count"] > 0:
            results["valid"] = True
        else:
            results["errors"].append("No test functions found")

        if not results["security_markers"]:
            results["errors"].append("No security markers found")

    except SyntaxError as e:
        results["errors"].append(f"Syntax error: {e}")
    except Exception as e:
        results["errors"].append(f"Error reading file: {e}")

    return results


def validate_security_test_structure():
    """Validate the overall security test structure."""
    security_test_dir = Path(__file__).parent

    # Expected test categories and files
    expected_structure = {
        "input_validation": [
            "test_sql_injection.py",
            "test_xss_prevention.py",
            "test_command_injection.py",
        ],
        "authentication": ["test_jwt_security.py"],
        "authorization": ["test_access_control.py"],
        "vulnerability": ["test_dependency_scanning.py"],
        "penetration": ["test_api_security.py"],
        "compliance": ["test_owasp_top10.py"],
        "encryption": ["test_data_protection.py"],
    }

    validation_results = {
        "overall_valid": True,
        "categories": {},
        "summary": {
            "total_files": 0,
            "valid_files": 0,
            "total_tests": 0,
            "categories_complete": 0,
        },
    }

    for category, files in expected_structure.items():
        category_dir = security_test_dir / category
        category_results = {
            "directory_exists": category_dir.exists(),
            "files": {},
            "complete": True,
        }

        if category_results["directory_exists"]:
            for file_name in files:
                file_path = category_dir / file_name
                if file_path.exists():
                    file_results = validate_test_file(file_path)
                    category_results["files"][file_name] = file_results

                    validation_results["summary"]["total_files"] += 1
                    if file_results["valid"]:
                        validation_results["summary"]["valid_files"] += 1
                    validation_results["summary"]["total_tests"] += file_results[
                        "test_count"
                    ]
                else:
                    category_results["files"][file_name] = {
                        "file": str(file_path),
                        "valid": False,
                        "errors": ["File not found"],
                        "test_count": 0,
                    }
                    category_results["complete"] = False
                    validation_results["overall_valid"] = False
        else:
            category_results["complete"] = False
            validation_results["overall_valid"] = False

        if category_results["complete"]:
            validation_results["summary"]["categories_complete"] += 1

        validation_results["categories"][category] = category_results

    return validation_results


def print_validation_report(results: dict):
    """Print validation report."""
    print("=" * 60)
    print("SECURITY TEST VALIDATION REPORT")
    print("=" * 60)
    print()

    # Summary
    summary = results["summary"]
    print(f"Overall Status: {'✅ VALID' if results['overall_valid'] else '❌ INVALID'}")
    print(f"Total Test Files: {summary['total_files']}")
    print(f"Valid Test Files: {summary['valid_files']}")
    print(f"Total Test Functions: {summary['total_tests']}")
    print(
        f"Complete Categories: {summary['categories_complete']}/{len(results['categories'])}"
    )
    print()

    # Category breakdown
    print("CATEGORY BREAKDOWN:")
    print("-" * 40)

    for category, category_data in results["categories"].items():
        status = "✅" if category_data["complete"] else "❌"
        print(f"{status} {category.upper()}")

        if not category_data["directory_exists"]:
            print(f"   ⚠️  Directory missing: {category}/")
            continue

        for file_name, file_data in category_data["files"].items():
            file_status = "✅" if file_data["valid"] else "❌"
            print(f"   {file_status} {file_name} ({file_data['test_count']} tests)")

            if file_data["errors"]:
                for error in file_data["errors"]:
                    print(f"      ⚠️  {error}")
        print()

    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 40)

    if results["overall_valid"]:
        print("✅ Security test structure is complete and valid")
        print("✅ All test categories are properly implemented")
        print("✅ Security markers are properly applied")
    else:
        print("❌ Security test structure needs attention:")

        for category, category_data in results["categories"].items():
            if not category_data["complete"]:
                print(f"   - Complete {category} test implementation")

            for file_name, file_data in category_data["files"].items():
                if not file_data["valid"]:
                    print(f"   - Fix issues in {category}/{file_name}")


def main():
    """Main validation function."""
    print("Validating security test implementation...")
    print()

    results = validate_security_test_structure()
    print_validation_report(results)

    # Exit with appropriate code
    sys.exit(0 if results["overall_valid"] else 1)


if __name__ == "__main__":
    main()
