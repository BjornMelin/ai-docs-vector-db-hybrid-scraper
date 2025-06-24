#!/usr/bin/env python3
"""Test contract framework validation script.

This script validates that the contract testing framework is properly set up
and all components are working correctly.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_contract_directory_structure():
    """Test that contract test directory structure exists."""
    contract_dir = project_root / "tests" / "contract"
    
    expected_dirs = [
        "api_contracts",
        "schema_validation", 
        "pact",
        "openapi",
        "consumer_driven"
    ]
    
    expected_files = [
        "conftest.py",
        "README.md",
        "__init__.py"
    ]
    
    print("Testing contract directory structure...")
    
    # Check main directory exists
    assert contract_dir.exists(), f"Contract directory not found: {contract_dir}"
    print(f"✓ Contract directory exists: {contract_dir}")
    
    # Check subdirectories
    for subdir in expected_dirs:
        subdir_path = contract_dir / subdir
        assert subdir_path.exists(), f"Contract subdirectory not found: {subdir_path}"
        print(f"✓ Subdirectory exists: {subdir}")
        
        # Check __init__.py exists
        init_file = subdir_path / "__init__.py"
        assert init_file.exists(), f"__init__.py not found in {subdir_path}"
    
    # Check main files
    for file_name in expected_files:
        file_path = contract_dir / file_name
        assert file_path.exists(), f"Contract file not found: {file_path}"
        print(f"✓ File exists: {file_name}")
    
    print("✓ All contract directory structure validated\n")


def test_contract_test_files():
    """Test that contract test files exist and have content."""
    contract_dir = project_root / "tests" / "contract"
    
    expected_test_files = [
        "openapi/test_spec_validation.py",
        "schema_validation/test_json_schemas.py",
        "pact/test_consumer_contracts.py",
        "api_contracts/test_endpoint_contracts.py",
        "consumer_driven/test_mcp_contracts.py",
        "test_contract_integration.py",
        "test_contract_runner.py",
        "test_framework_validation.py"
    ]
    
    print("Testing contract test files...")
    
    for test_file in expected_test_files:
        file_path = contract_dir / test_file
        assert file_path.exists(), f"Contract test file not found: {file_path}"
        
        # Check file has content
        content = file_path.read_text()
        assert len(content) > 100, f"Contract test file appears empty: {file_path}"
        assert "pytest" in content or "test_" in content, f"File doesn't appear to be a test file: {file_path}"
        
        print(f"✓ Test file validated: {test_file}")
    
    print("✓ All contract test files validated\n")


def test_contract_fixtures():
    """Test that contract fixtures are properly defined."""
    conftest_path = project_root / "tests" / "contract" / "conftest.py"
    
    print("Testing contract fixtures...")
    
    # Read conftest.py content
    conftest_content = conftest_path.read_text()
    
    # Check for expected fixtures
    expected_fixtures = [
        "contract_test_config",
        "json_schema_validator", 
        "api_contract_validator",
        "openapi_contract_manager",
        "pact_contract_builder",
        "contract_test_data",
        "mock_contract_service"
    ]
    
    for fixture in expected_fixtures:
        assert f"def {fixture}" in conftest_content, f"Fixture not found in conftest.py: {fixture}"
        print(f"✓ Fixture defined: {fixture}")
    
    # Check for pytest markers
    expected_markers = [
        "contract",
        "api_contract", 
        "schema_validation",
        "pact",
        "openapi",
        "consumer_driven"
    ]
    
    for marker in expected_markers:
        marker_config = f'"markers", "{marker}:'
        assert marker_config in conftest_content, f"Pytest marker not configured: {marker}"
        print(f"✓ Pytest marker configured: {marker}")
    
    print("✓ All contract fixtures validated\n")


def test_api_contract_models():
    """Test that API contract models are available."""
    print("Testing API contract models...")
    
    try:
        from src.models.api_contracts import (
            SearchRequest,
            SearchResponse,
            DocumentRequest,
            ErrorResponse,
            HealthCheckResponse
        )
        
        # Test model instantiation
        search_req = SearchRequest(query="test")
        print(f"✓ SearchRequest model works: {search_req.query}")
        
        search_resp = SearchResponse(
            success=True,
            timestamp=datetime.now().timestamp()
        )
        print(f"✓ SearchResponse model works: {search_resp.success}")
        
        error_resp = ErrorResponse(
            success=False,
            timestamp=datetime.now().timestamp(),
            error="Test error"
        )
        print(f"✓ ErrorResponse model works: {error_resp.error}")
        
        print("✓ All API contract models validated\n")
        
    except ImportError as e:
        print(f"⚠ API contract models not available (expected without full install): {e}\n")


def test_contract_dependencies():
    """Test contract testing dependencies."""
    print("Testing contract dependencies availability...")
    
    dependencies = [
        ("json", "JSON support"),
        ("datetime", "Datetime support"),  
        ("pathlib", "Path support"),
        ("typing", "Type hints"),
        ("unittest.mock", "Mocking support")
    ]
    
    for dep, desc in dependencies:
        try:
            __import__(dep)
            print(f"✓ {desc}: {dep}")
        except ImportError:
            print(f"✗ Missing {desc}: {dep}")
    
    # Test optional dependencies
    optional_deps = [
        ("jsonschema", "JSON Schema validation"),
        ("pydantic", "Data validation"),
        ("httpx", "HTTP client"),
        ("schemathesis", "OpenAPI testing"),
        ("responses", "HTTP mocking")
    ]
    
    print("\nOptional contract testing dependencies:")
    for dep, desc in optional_deps:
        try:
            __import__(dep)
            print(f"✓ {desc}: {dep}")
        except ImportError:
            print(f"⚠ Not installed (install with contract group): {desc}")
    
    print()


def test_contract_documentation():
    """Test that contract documentation exists."""
    print("Testing contract documentation...")
    
    readme_path = project_root / "tests" / "contract" / "README.md"
    readme_content = readme_path.read_text()
    
    # Check for key documentation sections
    expected_sections = [
        "Contract Testing Suite",
        "Running Contract Tests", 
        "API Contract Testing",
        "Schema Validation",
        "Pact Testing",
        "OpenAPI Testing",
        "Tools and Frameworks"
    ]
    
    for section in expected_sections:
        assert section in readme_content, f"Documentation section missing: {section}"
        print(f"✓ Documentation section: {section}")
    
    print("✓ Contract documentation validated\n")


def main():
    """Run all contract framework validation tests."""
    print("🧪 Contract Testing Framework Validation")
    print("=" * 50)
    print()
    
    try:
        test_contract_directory_structure()
        test_contract_test_files()
        test_contract_fixtures()
        test_api_contract_models()
        test_contract_dependencies()
        test_contract_documentation()
        
        print("🎉 Contract Testing Framework Validation PASSED!")
        print()
        print("Next steps:")
        print("1. Install contract dependencies: uv sync --group contract")
        print("2. Run contract tests: uv run pytest tests/contract/ -v")
        print("3. Check specific contract types: uv run pytest -m contract")
        print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Contract Testing Framework Validation FAILED: {e}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())