#!/usr/bin/env python3
"""Simple configuration validation without external dependencies."""

import os
import sys
from pathlib import Path


def validate_project_structure():
    """Validate basic project structure."""
    print("📁 Validating project structure...")

    project_root = Path(__file__).parent.parent
    required_files = [
        "pyproject.toml",
        "src/__init__.py",
        "src/api/main.py",
        "src/cli/main.py",
        "src/cli/unified.py",
        "tests/conftest.py",
        ".vscode/settings.json",
        ".vscode/tasks.json",
    ]

    missing_files = []
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)

    return len(missing_files) == 0, missing_files


def validate_taskipy_integration():
    """Validate taskipy configuration."""
    print("🔧 Validating taskipy integration...")

    project_root = Path(__file__).parent.parent
    pyproject_path = project_root / "pyproject.toml"

    if not pyproject_path.exists():
        print("  ❌ pyproject.toml not found")
        return False

    content = pyproject_path.read_text()

    required_tasks = [
        "dev = ",
        "test = ",
        "quality = ",
        "docs-serve = ",
        "validate-config = "
    ]

    missing_tasks = []
    for task in required_tasks:
        if task in content:
            print(f"  ✅ Found task: {task.split('=')[0].strip()}")
        else:
            print(f"  ❌ Missing task: {task.split('=')[0].strip()}")
            missing_tasks.append(task)

    if "taskipy" in content:
        print("  ✅ taskipy dependency found")
    else:
        print("  ❌ taskipy dependency missing")
        missing_tasks.append("taskipy dependency")

    return len(missing_tasks) == 0, missing_tasks


def validate_scripts():
    """Validate that key scripts exist."""
    print("📜 Validating scripts...")

    project_root = Path(__file__).parent.parent
    scripts_dir = project_root / "scripts"

    required_scripts = [
        "run_fast_tests.py",
        "validate_config.py",
        "docs_automation.py",
    ]

    missing_scripts = []
    for script in required_scripts:
        script_path = scripts_dir / script
        if script_path.exists():
            print(f"  ✅ {script}")
        else:
            print(f"  ❌ {script}")
            missing_scripts.append(script)

    return len(missing_scripts) == 0, missing_scripts


def main():
    """Run simple validation."""
    print("🔍 Running simple project validation...\n")

    validations = [
        validate_project_structure,
        validate_taskipy_integration,
        validate_scripts
    ]

    all_passed = True
    issues = []

    for validation in validations:
        try:
            result, missing = validation()
            all_passed = all_passed and result
            if missing:
                issues.extend(missing)
        except Exception as e:
            print(f"  ❌ Validation error: {e}")
            all_passed = False
            issues.append(f"Validation error: {e}")
        print()

    print("📊 Validation Summary:")
    if all_passed:
        print("✅ All validations passed!")
        print("\n🚀 Developer Experience Optimization complete!")
        print("\nNext steps:")
        print("1. Install dependencies: uv sync")
        print("2. Run: task dev-simple")
        print("3. Test: task test")
        return 0
    print("❌ Validation failed!")
    if issues:
        print("\nIssues found:")
        for issue in issues:
            print(f"  • {issue}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
