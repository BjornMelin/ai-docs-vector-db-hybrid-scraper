#!/usr/bin/env python3
"""Security utilities for MCP server."""

import logging
import os
import re
from typing import ClassVar
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Security-related error."""

    pass


class SecurityValidator:
    """Security validation utilities."""

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

    @classmethod
    def validate_url(cls, url: str) -> str:
        """Validate and sanitize URL input.

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
        if parsed.scheme.lower() not in cls.ALLOWED_SCHEMES:
            raise SecurityError(f"URL scheme '{parsed.scheme}' not allowed")

        # Check for dangerous patterns
        url_lower = url.lower()
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, url_lower):
                raise SecurityError(f"URL contains dangerous pattern: {pattern}")

        # Basic length check
        if len(url) > 2048:
            raise SecurityError("URL too long (max 2048 characters)")

        return url.strip()

    @classmethod
    def validate_collection_name(cls, name: str) -> str:
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

    @classmethod
    def validate_query_string(cls, query: str) -> str:
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

        if len(query) > 1000:
            raise SecurityError("Query too long (max 1000 characters)")

        # Remove potentially dangerous characters
        query = re.sub(r'[<>"\']', "", query)

        return query

    @classmethod
    def sanitize_filename(cls, filename: str) -> str:
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


class APIKeyValidator:
    """API key validation and management."""

    @staticmethod
    def validate_openai_key(api_key: str) -> bool:
        """Validate OpenAI API key format.

        Args:
            api_key: API key to validate

        Returns:
            True if format is valid
        """
        if not api_key or not isinstance(api_key, str):
            return False

        # OpenAI API keys typically start with 'sk-' and are 51+ characters
        return api_key.startswith("sk-") and len(api_key) >= 51

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

    @staticmethod
    def validate_required_env_vars() -> dict[str, str]:
        """Validate required environment variables.

        Returns:
            Dictionary of validated environment variables

        Raises:
            SecurityError: If required variables are missing or invalid
        """
        required_vars = ["OPENAI_API_KEY"]
        optional_vars = ["QDRANT_URL", "FIRECRAWL_API_KEY"]

        env_vars = {}

        # Check required variables
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                raise SecurityError(f"Required environment variable {var} is not set")

            # Validate OpenAI API key format
            if var == "OPENAI_API_KEY" and not APIKeyValidator.validate_openai_key(
                value
            ):
                raise SecurityError(f"Invalid {var} format")

            env_vars[var] = value
            logger.info(f"{var}: {APIKeyValidator.mask_api_key(value)}")

        # Check optional variables
        for var in optional_vars:
            value = os.getenv(var)
            if value:
                env_vars[var] = value
                if "key" in var.lower():
                    logger.info(f"{var}: {APIKeyValidator.mask_api_key(value)}")
                else:
                    logger.info(f"{var}: {value}")
            else:
                logger.info(f"{var}: not set (optional)")

        return env_vars


def validate_startup_security() -> dict[str, str]:
    """Validate security requirements at startup.

    Returns:
        Validated environment variables

    Raises:
        SecurityError: If security validation fails
    """
    logger.info("Validating security requirements...")

    try:
        env_vars = APIKeyValidator.validate_required_env_vars()
        logger.info("✅ Security validation passed")
        return env_vars
    except SecurityError as e:
        logger.error(f"❌ Security validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during security validation: {e}")
        raise SecurityError(f"Security validation error: {e}") from e
