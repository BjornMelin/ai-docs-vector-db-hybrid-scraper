"""Tests for QueryValidationHelper."""

from unittest.mock import Mock, patch

import pytest

from src.mcp_tools.tools.helpers.validation_helper import QueryValidationHelper
from src.services.query_processing.models import MatryoshkaDimension, SearchStrategy


class MockContext:
    """Mock context for testing."""

    def __init__(self):
        self.logs = {"info": [], "debug": [], "warning": [], "error": []}

    async def info(self, msg: str):
        self.logs["info"].append(msg)

    async def debug(self, msg: str):
        self.logs["debug"].append(msg)

    async def warning(self, msg: str):
        self.logs["warning"].append(msg)

    async def error(self, msg: str):
        self.logs["error"].append(msg)


@pytest.fixture
def mock_context():
    """Create mock context."""
    return MockContext()


@pytest.fixture
def validation_helper():
    """Create validation helper with mocked security validator."""
    with patch(
        "src.mcp_tools.tools.helpers.validation_helper.SecurityValidator.from_unified_config"
    ) as mock_security:
        mock_validator = Mock()
        mock_security.return_value = mock_validator
        helper = QueryValidationHelper()
        helper.security_validator = mock_validator
        return helper


class TestQueryValidationHelper:
    """Test QueryValidationHelper functionality."""

    def test_initialization(self):
        """Test validation helper initialization."""
        with patch(
            "src.mcp_tools.tools.helpers.validation_helper.SecurityValidator.from_unified_config"
        ) as mock_security:
            mock_validator = Mock()
            mock_security.return_value = mock_validator

            helper = QueryValidationHelper()

            # Verify security validator was created
            mock_security.assert_called_once()
            assert helper.security_validator is mock_validator

    def test_validate_query_request_success(self, validation_helper):
        """Test successful query request validation."""
        # Setup mock request
        mock_request = Mock()
        mock_request.collection = "documentation"
        mock_request.query = "How to implement authentication?"

        # Setup mock validator responses
        validation_helper.security_validator.validate_collection_name.return_value = (
            "documentation"
        )
        validation_helper.security_validator.validate_query_string.return_value = (
            "How to implement authentication?"
        )

        # Validate request
        collection, query = validation_helper.validate_query_request(mock_request)

        # Verify validation calls
        validation_helper.security_validator.validate_collection_name.assert_called_once_with(
            "documentation"
        )
        validation_helper.security_validator.validate_query_string.assert_called_once_with(
            "How to implement authentication?"
        )

        # Verify results
        assert collection == "documentation"
        assert query == "How to implement authentication?"

    def test_validate_query_request_with_sanitization(self, validation_helper):
        """Test query request validation with sanitization."""
        # Setup mock request with potentially dangerous content
        mock_request = Mock()
        mock_request.collection = "docs; DROP TABLE users;"
        mock_request.query = "<script>alert('xss')</script>How to hack?"

        # Setup mock validator to sanitize
        validation_helper.security_validator.validate_collection_name.return_value = (
            "docs"
        )
        validation_helper.security_validator.validate_query_string.return_value = (
            "How to hack?"
        )

        # Validate request
        collection, query = validation_helper.validate_query_request(mock_request)

        # Verify sanitized results
        assert collection == "docs"
        assert query == "How to hack?"

    async def test_validate_force_options_valid_strategy(
        self, validation_helper, mock_context
    ):
        """Test validation of valid force strategy."""
        # Setup mock request
        mock_request = Mock()
        mock_request.force_strategy = "semantic"
        mock_request.force_dimension = None

        # Validate force options
        strategy, dimension = await validation_helper.validate_force_options(
            mock_request, mock_context
        )

        # Verify results
        assert strategy == SearchStrategy.SEMANTIC
        assert dimension is None
        assert len(mock_context.logs["warning"]) == 0

    async def test_validate_force_options_valid_dimension(
        self, validation_helper, mock_context
    ):
        """Test validation of valid force dimension."""
        # Setup mock request
        mock_request = Mock()
        mock_request.force_strategy = None
        mock_request.force_dimension = 1536

        # Validate force options
        strategy, dimension = await validation_helper.validate_force_options(
            mock_request, mock_context
        )

        # Verify results
        assert strategy is None
        assert dimension == MatryoshkaDimension.LARGE
        assert len(mock_context.logs["warning"]) == 0

    async def test_validate_force_options_all_valid(
        self, validation_helper, mock_context
    ):
        """Test validation of all valid force options."""
        # Setup mock request
        mock_request = Mock()
        mock_request.force_strategy = "hyde"
        mock_request.force_dimension = 512

        # Validate force options
        strategy, dimension = await validation_helper.validate_force_options(
            mock_request, mock_context
        )

        # Verify results
        assert strategy == SearchStrategy.HYDE
        assert dimension == MatryoshkaDimension.SMALL
        assert len(mock_context.logs["warning"]) == 0

    async def test_validate_force_options_invalid_strategy(
        self, validation_helper, mock_context
    ):
        """Test validation of invalid force strategy."""
        # Setup mock request
        mock_request = Mock()
        mock_request.force_strategy = "invalid_strategy"
        mock_request.force_dimension = None

        # Validate force options
        strategy, dimension = await validation_helper.validate_force_options(
            mock_request, mock_context
        )

        # Verify results
        assert strategy is None
        assert dimension is None

        # Verify warning was logged
        assert len(mock_context.logs["warning"]) == 1
        assert (
            "Invalid force_strategy 'invalid_strategy'"
            in mock_context.logs["warning"][0]
        )

    async def test_validate_force_options_invalid_dimension(
        self, validation_helper, mock_context
    ):
        """Test validation of invalid force dimension."""
        # Setup mock request
        mock_request = Mock()
        mock_request.force_strategy = None
        mock_request.force_dimension = 999

        # Validate force options
        strategy, dimension = await validation_helper.validate_force_options(
            mock_request, mock_context
        )

        # Verify results
        assert strategy is None
        assert dimension is None

        # Verify warning was logged
        assert len(mock_context.logs["warning"]) == 1
        assert "Invalid force_dimension '999'" in mock_context.logs["warning"][0]

    async def test_validate_force_options_dimension_exception(
        self, validation_helper, mock_context
    ):
        """Test validation when dimension processing raises exception."""
        # Setup mock request with non-integer dimension
        mock_request = Mock()
        mock_request.force_strategy = None
        mock_request.force_dimension = "not_a_number"

        # Validate force options
        strategy, dimension = await validation_helper.validate_force_options(
            mock_request, mock_context
        )

        # Verify results
        assert strategy is None
        assert dimension is None

        # Verify warning was logged for exception
        assert len(mock_context.logs["warning"]) == 1
        assert (
            "Invalid force_dimension 'not_a_number'" in mock_context.logs["warning"][0]
        )

    async def test_validate_force_options_none_values(
        self, validation_helper, mock_context
    ):
        """Test validation with None force options."""
        # Setup mock request
        mock_request = Mock()
        mock_request.force_strategy = None
        mock_request.force_dimension = None

        # Validate force options
        strategy, dimension = await validation_helper.validate_force_options(
            mock_request, mock_context
        )

        # Verify results
        assert strategy is None
        assert dimension is None
        assert len(mock_context.logs["warning"]) == 0

    async def test_validate_force_options_empty_strings(
        self, validation_helper, mock_context
    ):
        """Test validation with empty string force options."""
        # Setup mock request
        mock_request = Mock()
        mock_request.force_strategy = ""
        mock_request.force_dimension = ""

        # Validate force options
        strategy, dimension = await validation_helper.validate_force_options(
            mock_request, mock_context
        )

        # Verify results (empty strings are falsy, so should be ignored)
        assert strategy is None
        assert dimension is None
        assert len(mock_context.logs["warning"]) == 0

    async def test_validate_force_options_case_insensitive_strategy(
        self, validation_helper, mock_context
    ):
        """Test validation with different case strategy."""
        # Setup mock request with uppercase strategy
        mock_request = Mock()
        mock_request.force_strategy = "HYBRID"
        mock_request.force_dimension = None

        # Validate force options
        strategy, dimension = await validation_helper.validate_force_options(
            mock_request, mock_context
        )

        # Verify results (should be converted to lowercase)
        assert strategy == SearchStrategy.HYBRID
        assert dimension is None
        assert len(mock_context.logs["warning"]) == 0

    async def test_validate_force_options_all_dimension_values(
        self, validation_helper, mock_context
    ):
        """Test validation with all valid dimension values."""
        valid_dimensions = [
            (512, MatryoshkaDimension.SMALL),
            (768, MatryoshkaDimension.MEDIUM),
            (1536, MatryoshkaDimension.LARGE),
        ]

        for input_dim, expected_dim in valid_dimensions:
            # Setup mock request
            mock_request = Mock()
            mock_request.force_strategy = None
            mock_request.force_dimension = input_dim

            # Reset context logs
            mock_context.logs = {"info": [], "debug": [], "warning": [], "error": []}

            # Validate force options
            strategy, dimension = await validation_helper.validate_force_options(
                mock_request, mock_context
            )

            # Verify results
            assert strategy is None
            assert dimension == expected_dim
            assert len(mock_context.logs["warning"]) == 0

    async def test_validate_force_options_all_strategy_values(
        self, validation_helper, mock_context
    ):
        """Test validation with all valid strategy values."""
        valid_strategies = [
            ("semantic", SearchStrategy.SEMANTIC),
            ("hyde", SearchStrategy.HYDE),
            ("hybrid", SearchStrategy.HYBRID),
            ("multi_stage", SearchStrategy.MULTI_STAGE),
            ("reranked", SearchStrategy.RERANKED),
        ]

        for input_strategy, expected_strategy in valid_strategies:
            # Setup mock request
            mock_request = Mock()
            mock_request.force_strategy = input_strategy
            mock_request.force_dimension = None

            # Reset context logs
            mock_context.logs = {"info": [], "debug": [], "warning": [], "error": []}

            # Validate force options
            strategy, dimension = await validation_helper.validate_force_options(
                mock_request, mock_context
            )

            # Verify results
            assert strategy == expected_strategy
            assert dimension is None
            assert len(mock_context.logs["warning"]) == 0
