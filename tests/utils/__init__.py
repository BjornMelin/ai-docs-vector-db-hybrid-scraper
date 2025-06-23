"""Test utilities package.

This package provides reusable utilities for testing including assertion helpers,
data generators, mock factories, test configuration helpers, and performance
measurement utilities.
"""

# Make utils a proper Python package
__version__ = "1.0.0"

# Import key utilities for easy access
from .assertion_helpers import AssertionHelpers
from .assertion_helpers import assert_error_response
from .assertion_helpers import assert_pagination_response
from .assertion_helpers import assert_valid_response
from .data_generators import HypothesisStrategies
from .data_generators import TestDataGenerator
from .data_generators import generate_search_queries
from .data_generators import generate_test_documents
from .mock_factories import MockFactory
from .mock_factories import create_mock_embedding_service
from .mock_factories import create_mock_vector_db
from .mock_factories import create_mock_web_scraper
from .performance_utils import BenchmarkSuite
from .performance_utils import PerformanceTracker
from .performance_utils import measure_execution_time
from .performance_utils import memory_profiler
from .test_config_helpers import ConfigManager
from .test_config_helpers import cleanup_test_data
from .test_config_helpers import get_test_environment
from .test_config_helpers import setup_test_database

__all__ = [
    "AssertionHelpers",
    "BenchmarkSuite",
    # Test configuration helpers
    "ConfigManager",
    "HypothesisStrategies",
    # Mock factories
    "MockFactory",
    # Performance utilities
    "PerformanceTracker",
    # Data generators
    "TestDataGenerator",
    "assert_error_response",
    "assert_pagination_response",
    # Assertion helpers
    "assert_valid_response",
    "cleanup_test_data",
    "create_mock_embedding_service",
    "create_mock_vector_db",
    "create_mock_web_scraper",
    "generate_search_queries",
    "generate_test_documents",
    "get_test_environment",
    "measure_execution_time",
    "memory_profiler",
    "setup_test_database",
]
