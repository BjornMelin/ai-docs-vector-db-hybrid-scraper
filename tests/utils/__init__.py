"""Test utilities package.

This package provides reusable utilities for testing including assertion helpers,
data generators, mock factories, test configuration helpers, and performance
measurement utilities.
"""

# Make utils a proper Python package
__version__ = "1.0.0"

# Import key utilities for easy access
from .assertion_helpers import (
    assert_valid_response,
    assert_error_response,
    assert_pagination_response,
    AssertionHelpers,
)
from .data_generators import (
    TestDataGenerator,
    HypothesisStrategies,
    generate_test_documents,
    generate_search_queries,
)
from .mock_factories import (
    MockFactory,
    create_mock_vector_db,
    create_mock_embedding_service,
    create_mock_web_scraper,
)
from .performance_utils import (
    PerformanceTracker,
    measure_execution_time,
    memory_profiler,
    BenchmarkSuite,
)
from .test_config_helpers import (
    ConfigManager,
    get_test_environment,
    setup_test_database,
    cleanup_test_data,
)

__all__ = [
    # Assertion helpers
    "assert_valid_response",
    "assert_error_response", 
    "assert_pagination_response",
    "AssertionHelpers",
    
    # Data generators
    "TestDataGenerator",
    "HypothesisStrategies",
    "generate_test_documents",
    "generate_search_queries",
    
    # Mock factories
    "MockFactory",
    "create_mock_vector_db",
    "create_mock_embedding_service",
    "create_mock_web_scraper",
    
    # Performance utilities
    "PerformanceTracker",
    "measure_execution_time",
    "memory_profiler",
    "BenchmarkSuite",
    
    # Test configuration helpers
    "ConfigManager",
    "get_test_environment",
    "setup_test_database",
    "cleanup_test_data",
]