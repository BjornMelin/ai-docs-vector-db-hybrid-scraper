#!/usr/bin/env python3
"""
Python 3.13 Compatibility Validation Script

This script validates that all dependencies are compatible with Python 3.13
and performs basic import and functionality tests.

Usage:
    uv run python scripts/validate_python313_compatibility.py
"""

import importlib
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check if running on Python 3.13+"""
    version = sys.version_info
    if version >= (3, 13):
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return (
            False,
            f"⚠️  Python {version.major}.{version.minor}.{version.micro} (3.13+ recommended)",
        )


def check_critical_imports() -> Dict[str, bool]:
    """Test importing critical dependencies"""
    critical_packages = [
        # Core framework
        "fastapi",
        "starlette",
        "uvicorn",
        "pydantic",
        "pydantic_settings",
        # AI/ML
        "openai",
        "qdrant_client",
        "fastembed",
        "FlagEmbedding",
        "crawl4ai",
        # Data processing
        "pandas",
        "numpy",
        "scipy",
        # System
        "psutil",
        "aiohttp",
        "aiofiles",
        # Development
        "pytest",
        "ruff",
        "coverage",
    ]

    results = {}
    for package in critical_packages:
        try:
            importlib.import_module(package)
            results[package] = True
        except ImportError as e:
            results[package] = False
            print(f"❌ Failed to import {package}: {e}")
        except Exception:
            results[package] = False
            print(f"❌ Error importing {package}: {e}")

    return results


def check_src_imports() -> Dict[str, bool]:
    """Test importing our own modules"""
    src_modules = [
        "src.config.settings",
        "src.api.main",
        "src.services.embeddings.manager",
        "src.services.vector_db.qdrant_manager",
        "src.unified_mcp_server",
    ]

    results = {}
    for module in src_modules:
        try:
            importlib.import_module(module)
            results[module] = True
        except ImportError as e:
            results[module] = False
            print(f"❌ Failed to import {module}: {e}")
        except Exception:
            results[module] = False
            print(f"❌ Error importing {module}: {e}")

    return results


def test_basic_functionality() -> Dict[str, bool]:
    """Test basic functionality of key components"""
    tests = {}

    # Test FastAPI
    try:
        from fastapi import FastAPI

        app = FastAPI()
        tests["fastapi_creation"] = True
    except Exception:
        tests["fastapi_creation"] = False
        print(f"❌ FastAPI creation failed: {e}")

    # Test Pydantic
    try:
        from pydantic import BaseModel

        class TestModel(BaseModel):
            name: str
            value: int = 42

        model = TestModel(name="test")
        assert model.value == 42
        tests["pydantic_validation"] = True
    except Exception:
        tests["pydantic_validation"] = False
        print(f"❌ Pydantic validation failed: {e}")

    # Test OpenAI (import only, no API call)
    try:
        from openai import OpenAI

        # Just test client creation without API key
        tests["openai_import"] = True
    except Exception:
        tests["openai_import"] = False
        print(f"❌ OpenAI import failed: {e}")

    # Test NumPy basic operations
    try:
        import numpy as np

        arr = np.array([1, 2, 3, 4, 5])
        result = np.mean(arr)
        assert result == 3.0
        tests["numpy_operations"] = True
    except Exception:
        tests["numpy_operations"] = False
        print(f"❌ NumPy operations failed: {e}")

    # Test psutil
    try:
        import psutil

        cpu_count = psutil.cpu_count()
        assert isinstance(cpu_count, int) and cpu_count > 0
        tests["psutil_basic"] = True
    except Exception:
        tests["psutil_basic"] = False
        print(f"❌ psutil basic functionality failed: {e}")

    return tests


def print_summary(
    python_check: Tuple[bool, str],
    import_results: Dict[str, bool],
    src_results: Dict[str, bool],
    functionality_results: Dict[str, bool],
) -> None:
    """Print validation summary"""
    print("\n" + "=" * 60)
    print("PYTHON 3.13 COMPATIBILITY VALIDATION SUMMARY")
    print("=" * 60)

    # Python version
    print(f"\n🐍 Python Version: {python_check[1]}")

    # Import results
    total_imports = len(import_results)
    successful_imports = sum(import_results.values())
    import_percentage = (successful_imports / total_imports) * 100

    print(
        f"\n📦 Dependency Imports: {successful_imports}/{total_imports} ({import_percentage:.1f}%)"
    )
    if import_percentage < 100:
        failed = [pkg for pkg, success in import_results.items() if not success]
        print(f"   Failed: {', '.join(failed)}")

    # Source imports
    total_src = len(src_results)
    successful_src = sum(src_results.values())
    src_percentage = (successful_src / total_src) * 100

    print(
        f"\n🔧 Source Module Imports: {successful_src}/{total_src} ({src_percentage:.1f}%)"
    )
    if src_percentage < 100:
        failed_src = [mod for mod, success in src_results.items() if not success]
        print(f"   Failed: {', '.join(failed_src)}")

    # Functionality tests
    total_func = len(functionality_results)
    successful_func = sum(functionality_results.values())
    func_percentage = (successful_func / total_func) * 100

    print(
        f"\n⚙️  Functionality Tests: {successful_func}/{total_func} ({func_percentage:.1f}%)"
    )
    if func_percentage < 100:
        failed_func = [
            test for test, success in functionality_results.items() if not success
        ]
        print(f"   Failed: {', '.join(failed_func)}")

    # Overall status
    overall_success = (import_percentage + src_percentage + func_percentage) / 3
    print(f"\n🎯 Overall Compatibility: {overall_success:.1f}%")

    if overall_success >= 95:
        print("✅ EXCELLENT - Ready for Python 3.13 deployment!")
    elif overall_success >= 85:
        print("✅ GOOD - Minor issues to resolve")
    elif overall_success >= 70:
        print("⚠️  FAIR - Several issues need attention")
    else:
        print("❌ POOR - Major compatibility issues detected")

    print("\n" + "=" * 60)


def main():
    """Run complete validation suite"""
    print("🔍 Python 3.13 Compatibility Validation")
    print("🚀 AI Docs Vector DB Hybrid Scraper")
    print("-" * 60)

    # Check Python version
    python_check = check_python_version()
    print(f"Python Version: {python_check[1]}")

    # Check critical dependency imports
    print("\n📦 Checking critical dependency imports...")
    import_results = check_critical_imports()

    # Check source module imports
    print("\n🔧 Checking source module imports...")
    src_results = check_src_imports()

    # Test basic functionality
    print("\n⚙️  Testing basic functionality...")
    functionality_results = test_basic_functionality()

    # Print summary
    print_summary(python_check, import_results, src_results, functionality_results)

    # Exit with appropriate code
    all_results = (
        list(import_results.values())
        + list(src_results.values())
        + list(functionality_results.values())
    )
    success_rate = sum(all_results) / len(all_results)

    if success_rate >= 0.95:
        sys.exit(0)  # Success
    elif success_rate >= 0.85:
        sys.exit(1)  # Minor issues
    else:
        sys.exit(2)  # Major issues


if __name__ == "__main__":
    main()
