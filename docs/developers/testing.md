# Unified Testing Guide

## Test Categories

# Unit Tests - Test individual functions/classes in isolation
# Integration Tests - Test interactions between components
# End-to-End Tests - Test complete user workflows

## Essential Test Commands

# Run all tests
# pytest

# Run tests in specific file
# pytest tests/test_example.py

# Run tests with verbose output
# pytest -v

# Run tests matching pattern
# pytest -k "test_function_name"

# Run tests with coverage report
# pytest --cov=src --cov-report=term-missing

# Run tests in parallel
# pytest -n auto

# Run tests with specific marker
# pytest -m integration

## Test Structure and Patterns

import pytest
from unittest.mock import Mock, patch, AsyncMock

# AAA Pattern: Arrange, Act, Assert
def test_example():
    # Arrange
    input_data = "test"
    expected = "TEST"
    
    # Act
    result = input_data.upper()
    
    # Assert
    assert result == expected

# Fixtures
@pytest.fixture
def sample_data():
    return {"key": "value"}

@pytest.fixture(scope="session")
def database_connection():
    # Setup
    conn = connect_to_db()
    yield conn
    # Teardown
    conn.close()

def test_with_fixture(sample_data):
    assert sample_data["key"] == "value"

## Async Testing Basics

import asyncio

@pytest.mark.asyncio
async def test_async_function():
    result = await async_function()
    assert result is not None

# Async fixtures
@pytest.fixture
async def async_resource():
    resource = await create_async_resource()
    yield resource
    await resource.cleanup()

## Mocking Patterns

# Mock external service calls
@patch("src.service.external_api_call")
def test_with_mock(mock_api):
    mock_api.return_value = {"status": "success"}
    result = function_that_calls_api()
    assert result["status"] == "success"

# Mock async functions
@pytest.mark.asyncio
@patch("src.service.async_external_call", new_callable=AsyncMock)
async def test_async_mock(mock_call):
    mock_call.return_value = "mocked_response"
    result = await function_that_calls_async_api()
    assert result == "mocked_response"

# Mock context manager
def test_with_context_manager():
    mock_file = Mock()
    mock_file.__enter__ = Mock(return_value=mock_file)
    mock_file.read = Mock(return_value="file_content")
    
    with patch("builtins.open", return_value=mock_file):
        content = read_file("test.txt")
        assert content == "file_content"

## Coverage Requirements

# Minimum coverage threshold
# pytest --cov=src --cov-fail-under=80

# Coverage configuration in pyproject.toml
# [tool.coverage.run]
# branch = true
# source = ["src"]

# [tool.coverage.report]
# exclude_lines = ["pragma: no cover", "def __repr__"]

## Test Organization and Markers

# Directory structure:
# tests/
#   unit/
#   integration/
#   e2e/

# Custom markers in pytest.ini or pyproject.toml
# markers = [
#     "unit: unit tests",
#     "integration: integration tests", 
#     "e2e: end-to-end tests",
#     "slow: slow running tests"
# ]

@pytest.mark.unit
def test_unit_function():
    pass

@pytest.mark.integration
def test_integration_flow():
    pass

@pytest.mark.e2e
def test_user_workflow():
    pass

@pytest.mark.slow
def test_long_running_process():
    pass