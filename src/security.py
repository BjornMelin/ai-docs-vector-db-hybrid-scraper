#!/usr/bin/env python3
"""Security utilities for MCP server with unified configuration integration."""

import logging
import os
import re
from typing import ClassVar
from urllib.parse import urlparse

from src.config import SecurityConfig, get_config


logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Security-related error."""

    pass


class SecurityValidator:
    """Security validation utilities with unified configuration integration."""

    # Allowed URL schemes
    ALLOWED_SCHEMES: ClassVar[set[str]] = {"http", "https"}

    # Dangerous patterns to block
    DANGEROUS_PATTERNS: ClassVar[list[str]] = [
        r"javascript:",
        r"data:",
        r"file:",
        r"ftp:",
        r"localhost",
        r"127\.0\.0\.1",
        r"0\.0\.0\.0",
        r"::1",
        # Block common internal network ranges
        r"192\.168\.",
        r"10\.",
        r"172\.(1[6-9]|2[0-9]|3[0-1])\.",
    ]

    def __init__(self, security_config: SecurityConfig | None = None):
        """Initialize with security configuration.

        Args:
            security_config: Security configuration. If None, loads from unified config.
        """
        self.config = security_config or get_config().security
        logger.info(
            f"SecurityValidator initialized with {len(self.config.allowed_domains)} allowed domains"
        )

    @classmethod
    def from_unified_config(cls) -> "SecurityValidator":
        """Create SecurityValidator from unified configuration."""
        return cls(get_config().security)

    def validate_url(self, url: str) -> str:
        """Validate and sanitize URL input using unified configuration.

        Args:
            url: URL to validate

        Returns:
            Sanitized URL

        Raises:
            SecurityError: If URL is potentially dangerous
        """
        if not url or not isinstance(url, str):
            raise SecurityError("URL must be a non-empty string")

        # Parse URL
        try:
            parsed = urlparse(url.strip())
        except Exception as e:
            raise SecurityError(f"Invalid URL format: {e}") from e

        # Check scheme
        if parsed.scheme.lower() not in self.ALLOWED_SCHEMES:
            raise SecurityError(f"URL scheme '{parsed.scheme}' not allowed")

        # Check against blocked domains from config
        domain = parsed.netloc.lower()
        for blocked in self.config.blocked_domains:
            if blocked.lower() in domain:
                raise SecurityError(f"Domain '{domain}' is blocked")

        # Check against allowed domains if configured
        if self.config.allowed_domains:
            domain_allowed = False
            for allowed in self.config.allowed_domains:
                if allowed.lower() in domain or domain in allowed.lower():
                    domain_allowed = True
                    break
            if not domain_allowed:
                raise SecurityError(f"Domain '{domain}' not in allowed list")

        # Check for dangerous patterns
        url_lower = url.lower()
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, url_lower):
                raise SecurityError(f"URL contains dangerous pattern: {pattern}")

        # Basic length check
        if len(url) > 2048:
            raise SecurityError("URL too long (max 2048 characters)")

        return url.strip()

    @classmethod
    def validate_url_static(cls, url: str) -> str:
        """Static method for backward compatibility - uses default config."""
        validator = cls.from_unified_config()
        return validator.validate_url(url)

    def validate_collection_name(self, name: str) -> str:
        """Validate collection name.

        Args:
            name: Collection name to validate

        Returns:
            Sanitized collection name

        Raises:
            SecurityError: If name is invalid
        """
        if not name or not isinstance(name, str):
            raise SecurityError("Collection name must be a non-empty string")

        # Strip and check length
        name = name.strip()
        if len(name) < 1:
            raise SecurityError("Collection name cannot be empty")

        if len(name) > 64:
            raise SecurityError("Collection name too long (max 64 characters)")

        # Check for valid characters (alphanumeric, underscore, hyphen)
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise SecurityError(
                "Collection name can only contain letters, numbers, underscore, and hyphen"
            )

        return name

    def validate_query_string(self, query: str) -> str:
        """Validate search query string.

        Args:
            query: Search query to validate

        Returns:
            Sanitized query

        Raises:
            SecurityError: If query is invalid
        """
        if not query or not isinstance(query, str):
            raise SecurityError("Query must be a non-empty string")

        # Strip and check length
        query = query.strip()
        if len(query) < 1:
            raise SecurityError("Query cannot be empty")

        # Check max query length from config or use default
        max_length = getattr(self.config, "max_query_length", 1000)
        if len(query) > max_length:
            raise SecurityError(f"Query too long (max {max_length} characters)")

        # Remove potentially dangerous characters
        query = re.sub(r'[<>"\']', "", query)

        return query

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file operations.

        Args:
            filename: Filename to sanitize

        Returns:
            Sanitized filename
        """
        if not filename or not isinstance(filename, str):
            return "safe_filename"

        # Remove path traversal attempts
        filename = os.path.basename(filename.strip())

        # Remove dangerous characters
        filename = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)

        # Limit length
        if len(filename) > 255:
            filename = filename[:255]

        # Ensure it's not empty
        if not filename:
            filename = "safe_filename"

        return filename

    @classmethod
    def validate_collection_name_static(cls, name: str) -> str:
        """Static method for backward compatibility - uses default config."""
        validator = cls.from_unified_config()
        return validator.validate_collection_name(name)

    @classmethod
    def validate_query_string_static(cls, query: str) -> str:
        """Static method for backward compatibility - uses default config."""
        validator = cls.from_unified_config()
        return validator.validate_query_string(query)

    @classmethod
    def sanitize_filename_static(cls, filename: str) -> str:
        """Static method for backward compatibility - uses default config."""
        validator = cls.from_unified_config()
        return validator.sanitize_filename(filename)


class APIKeyValidator:
    """API key validation and management utilities."""

    @staticmethod
    def mask_api_key(api_key: str) -> str:
        """Mask API key for logging.

        Args:
            api_key: API key to mask

        Returns:
            Masked API key
        """
        if not api_key or len(api_key) < 8:
            return "*" * 12

        return f"{api_key[:4]}{'*' * 8}{api_key[-4:]}"
