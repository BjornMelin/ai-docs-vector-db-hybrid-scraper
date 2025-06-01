"""Unit tests for security module."""

from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.config import SecurityConfig
from src.security import APIKeyValidator
from src.security import SecurityError
from src.security import SecurityValidator


class TestSecurityError:
    """Test SecurityError exception."""

    def test_is_exception(self):
        """Test that SecurityError is an Exception."""
        error = SecurityError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"


class TestSecurityValidator:
    """Test SecurityValidator class."""

    @pytest.fixture
    def mock_security_config(self):
        """Create a mock security config."""
        config = Mock(spec=SecurityConfig)
        config.allowed_domains = []
        config.blocked_domains = []
        config.max_query_length = 1000
        return config

    @pytest.fixture
    def validator(self, mock_security_config):
        """Create a SecurityValidator instance with mock config."""
        return SecurityValidator(mock_security_config)

    def test_init_with_config(self, mock_security_config):
        """Test initialization with provided config."""
        validator = SecurityValidator(mock_security_config)
        assert validator.config == mock_security_config

    @patch("src.security.get_config")
    def test_init_without_config(self, mock_get_config):
        """Test initialization without config (loads from unified config)."""
        mock_config = Mock()
        mock_config.security = Mock(spec=SecurityConfig)
        mock_config.security.allowed_domains = []
        mock_config.security.blocked_domains = []
        mock_get_config.return_value = mock_config

        validator = SecurityValidator()
        assert validator.config == mock_config.security

    @patch("src.security.get_config")
    def test_from_unified_config(self, mock_get_config):
        """Test from_unified_config class method."""
        mock_config = Mock()
        mock_config.security = Mock(spec=SecurityConfig)
        mock_config.security.allowed_domains = []
        mock_config.security.blocked_domains = []
        mock_get_config.return_value = mock_config

        validator = SecurityValidator.from_unified_config()
        assert validator.config == mock_config.security

    def test_validate_url_valid(self, validator):
        """Test validation of valid URLs."""
        # HTTP URL
        assert validator.validate_url("http://example.com") == "http://example.com"

        # HTTPS URL
        assert validator.validate_url("https://example.com") == "https://example.com"

        # URL with path
        assert (
            validator.validate_url("https://example.com/path")
            == "https://example.com/path"
        )

        # URL with query params
        assert (
            validator.validate_url("https://example.com?q=test")
            == "https://example.com?q=test"
        )

        # URL with whitespace (gets stripped)
        assert (
            validator.validate_url("  https://example.com  ") == "https://example.com"
        )

    def test_validate_url_empty(self, validator):
        """Test validation of empty URL."""
        with pytest.raises(SecurityError) as exc_info:
            validator.validate_url("")
        assert "non-empty string" in str(exc_info.value)

        with pytest.raises(SecurityError):
            validator.validate_url(None)

    def test_validate_url_invalid_scheme(self, validator):
        """Test validation of URL with invalid scheme."""
        with pytest.raises(SecurityError) as exc_info:
            validator.validate_url("ftp://example.com")
        assert "scheme 'ftp' not allowed" in str(exc_info.value)

        with pytest.raises(SecurityError):
            validator.validate_url("javascript:alert('xss')")

        with pytest.raises(SecurityError):
            validator.validate_url("data:text/html,<script>alert('xss')</script>")

    def test_validate_url_blocked_domains(self, validator):
        """Test validation with blocked domains."""
        validator.config.blocked_domains = ["evil.com", "spam.net"]

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_url("https://evil.com")
        assert "blocked" in str(exc_info.value)

        with pytest.raises(SecurityError):
            validator.validate_url("https://subdomain.spam.net")

    def test_validate_url_allowed_domains(self, validator):
        """Test validation with allowed domains."""
        validator.config.allowed_domains = ["example.com", "trusted.org"]

        # Allowed domains
        assert validator.validate_url("https://example.com") == "https://example.com"
        assert (
            validator.validate_url("https://api.example.com")
            == "https://api.example.com"
        )
        assert validator.validate_url("https://trusted.org") == "https://trusted.org"

        # Not allowed
        with pytest.raises(SecurityError) as exc_info:
            validator.validate_url("https://other.com")
        assert "not in allowed list" in str(exc_info.value)

    def test_validate_url_dangerous_patterns(self, validator):
        """Test validation of URLs with dangerous patterns."""
        dangerous_urls = [
            "http://localhost",
            "http://127.0.0.1",
            "http://192.168.1.1",
            "http://10.0.0.1",
            "http://172.16.0.1",
            "http://::1",
            "http://0.0.0.0",
        ]

        for url in dangerous_urls:
            with pytest.raises(SecurityError) as exc_info:
                validator.validate_url(url)
            assert "dangerous pattern" in str(exc_info.value)

    def test_validate_url_too_long(self, validator):
        """Test validation of URL that's too long."""
        long_url = "https://example.com/" + "a" * 2048
        with pytest.raises(SecurityError) as exc_info:
            validator.validate_url(long_url)
        assert "too long" in str(exc_info.value)

    @patch("src.security.get_config")
    def test_validate_url_static(self, mock_get_config):
        """Test static validate_url method."""
        mock_config = Mock()
        mock_config.security = Mock(spec=SecurityConfig)
        mock_config.security.allowed_domains = []
        mock_config.security.blocked_domains = []
        mock_get_config.return_value = mock_config

        result = SecurityValidator.validate_url_static("https://example.com")
        assert result == "https://example.com"

    def test_validate_collection_name_valid(self, validator):
        """Test validation of valid collection names."""
        assert validator.validate_collection_name("documents") == "documents"
        assert (
            validator.validate_collection_name("test_collection") == "test_collection"
        )
        assert (
            validator.validate_collection_name("my-collection-123")
            == "my-collection-123"
        )
        assert (
            validator.validate_collection_name("  test  ") == "test"
        )  # Strips whitespace

    def test_validate_collection_name_empty(self, validator):
        """Test validation of empty collection name."""
        with pytest.raises(SecurityError) as exc_info:
            validator.validate_collection_name("")
        assert "non-empty string" in str(exc_info.value)

        with pytest.raises(SecurityError):
            validator.validate_collection_name(None)

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_collection_name("   ")
        assert "cannot be empty" in str(exc_info.value)

    def test_validate_collection_name_too_long(self, validator):
        """Test validation of collection name that's too long."""
        long_name = "a" * 65
        with pytest.raises(SecurityError) as exc_info:
            validator.validate_collection_name(long_name)
        assert "too long" in str(exc_info.value)

    def test_validate_collection_name_invalid_chars(self, validator):
        """Test validation of collection name with invalid characters."""
        invalid_names = [
            "my collection",  # Space
            "my.collection",  # Dot
            "my@collection",  # At sign
            "my/collection",  # Slash
            "my\\collection",  # Backslash
            "my$collection",  # Dollar sign
        ]

        for name in invalid_names:
            with pytest.raises(SecurityError) as exc_info:
                validator.validate_collection_name(name)
            assert "can only contain" in str(exc_info.value)

    @patch("src.security.get_config")
    def test_validate_collection_name_static(self, mock_get_config):
        """Test static validate_collection_name method."""
        mock_config = Mock()
        mock_config.security = Mock(spec=SecurityConfig)
        mock_config.security.allowed_domains = []
        mock_config.security.blocked_domains = []
        mock_get_config.return_value = mock_config

        result = SecurityValidator.validate_collection_name_static("test")
        assert result == "test"

    def test_validate_query_string_valid(self, validator):
        """Test validation of valid query strings."""
        assert validator.validate_query_string("search term") == "search term"
        assert (
            validator.validate_query_string("  test  ") == "test"
        )  # Strips whitespace
        assert validator.validate_query_string("machine learning") == "machine learning"

    def test_validate_query_string_empty(self, validator):
        """Test validation of empty query string."""
        with pytest.raises(SecurityError) as exc_info:
            validator.validate_query_string("")
        assert "non-empty string" in str(exc_info.value)

        with pytest.raises(SecurityError):
            validator.validate_query_string(None)

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_query_string("   ")
        assert "cannot be empty" in str(exc_info.value)

    def test_validate_query_string_too_long(self, validator):
        """Test validation of query string that's too long."""
        validator.config.max_query_length = 100
        long_query = "a" * 101

        with pytest.raises(SecurityError) as exc_info:
            validator.validate_query_string(long_query)
        assert "too long" in str(exc_info.value)
        assert "max 100" in str(exc_info.value)

    def test_validate_query_string_dangerous_chars_removed(self, validator):
        """Test that dangerous characters are removed from query."""
        # Characters like <, >, ", ' are removed
        assert validator.validate_query_string("test <script>") == "test script"
        assert validator.validate_query_string("test > value") == "test  value"
        assert validator.validate_query_string('test "quoted"') == "test quoted"
        assert validator.validate_query_string("test's") == "tests"

    @patch("src.security.get_config")
    def test_validate_query_string_static(self, mock_get_config):
        """Test static validate_query_string method."""
        mock_config = Mock()
        mock_config.security = Mock(spec=SecurityConfig)
        mock_config.security.allowed_domains = []
        mock_config.security.blocked_domains = []
        mock_config.security.max_query_length = 1000
        mock_get_config.return_value = mock_config

        result = SecurityValidator.validate_query_string_static("test query")
        assert result == "test query"

    def test_sanitize_filename_valid(self, validator):
        """Test sanitization of valid filenames."""
        assert validator.sanitize_filename("document.pdf") == "document.pdf"
        assert validator.sanitize_filename("my-file_123.txt") == "my-file_123.txt"
        assert (
            validator.sanitize_filename("  test.doc  ") == "test.doc"
        )  # Strips whitespace

    def test_sanitize_filename_empty(self, validator):
        """Test sanitization of empty filename."""
        assert validator.sanitize_filename("") == "safe_filename"
        assert validator.sanitize_filename(None) == "safe_filename"
        assert validator.sanitize_filename("   ") == "safe_filename"

    def test_sanitize_filename_path_traversal(self, validator):
        """Test that path traversal attempts are removed."""
        assert validator.sanitize_filename("../../../etc/passwd") == "passwd"
        assert validator.sanitize_filename("/etc/passwd") == "passwd"
        # Windows-style path with backslashes get replaced with underscores
        assert (
            validator.sanitize_filename("C:\\Windows\\System32\\config")
            == "C__Windows_System32_config"
        )

    def test_sanitize_filename_dangerous_chars(self, validator):
        """Test that dangerous characters are replaced."""
        assert validator.sanitize_filename("file<name>.txt") == "file_name_.txt"
        assert validator.sanitize_filename("file:name|test") == "file_name_test"
        assert validator.sanitize_filename('file"name"') == "file_name_"
        assert validator.sanitize_filename("file?name*") == "file_name_"

    def test_sanitize_filename_too_long(self, validator):
        """Test that long filenames are truncated."""
        long_name = "a" * 300 + ".txt"
        result = validator.sanitize_filename(long_name)
        assert len(result) == 255
        assert result.startswith("aaa")

    def test_sanitize_filename_control_chars(self, validator):
        """Test that control characters are replaced."""
        # Control characters (0x00 - 0x1f) should be replaced with _
        filename = "file\x00\x01\x02.txt"
        result = validator.sanitize_filename(filename)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x02" not in result

    @patch("src.security.get_config")
    def test_sanitize_filename_static(self, mock_get_config):
        """Test static sanitize_filename method."""
        mock_config = Mock()
        mock_config.security = Mock(spec=SecurityConfig)
        mock_config.security.allowed_domains = []
        mock_config.security.blocked_domains = []
        mock_get_config.return_value = mock_config

        result = SecurityValidator.sanitize_filename_static("test.pdf")
        assert result == "test.pdf"


class TestAPIKeyValidator:
    """Test APIKeyValidator class."""

    def test_mask_api_key_normal(self):
        """Test masking of normal API keys."""
        # Normal key
        masked = APIKeyValidator.mask_api_key("sk-1234567890abcdef")
        assert masked == "sk-1********cdef"
        assert len(masked) == 16

        # Longer key
        masked = APIKeyValidator.mask_api_key("fc-abcdefghijklmnopqrstuvwxyz")
        assert masked == "fc-a********wxyz"

    def test_mask_api_key_short(self):
        """Test masking of short API keys."""
        # Too short (< 8 chars)
        masked = APIKeyValidator.mask_api_key("sk-123")
        assert masked == "************"

        # Empty
        masked = APIKeyValidator.mask_api_key("")
        assert masked == "************"

        # None
        masked = APIKeyValidator.mask_api_key(None)
        assert masked == "************"

    def test_mask_api_key_exactly_8(self):
        """Test masking of API key with exactly 8 characters."""
        masked = APIKeyValidator.mask_api_key("12345678")
        assert masked == "1234********5678"

    def test_mask_preserves_prefix_suffix(self):
        """Test that masking preserves prefix and suffix."""
        # Various prefixes
        assert APIKeyValidator.mask_api_key("sk-prod-1234567890").startswith("sk-p")
        assert APIKeyValidator.mask_api_key("sk-prod-1234567890").endswith("7890")

        assert APIKeyValidator.mask_api_key("fc-test-abcdefghij").startswith("fc-t")
        assert APIKeyValidator.mask_api_key("fc-test-abcdefghij").endswith("ghij")
