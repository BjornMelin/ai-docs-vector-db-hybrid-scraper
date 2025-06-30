#!/usr/bin/env python3
"""AI-specific security validation and protection mechanisms.

This module provides comprehensive security measures specifically designed
for AI and ML applications, including:

- Prompt injection detection and prevention
- Input validation for AI queries and documents
- Content sanitization for vector databases
- Metadata security validation
- Token limit enforcement
- Semantic attack detection
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional 

from fastapi import HTTPException

from src.config.security import SecurityConfig


logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels for security validation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityThreat:
    """Represents a detected security threat."""

    threat_type: str
    level: ThreatLevel
    description: str
    pattern_matched: str
    suggested_action: str


class AISecurityValidator:
    """Comprehensive security validation for AI-specific inputs and operations.

    This class provides robust security measures against various AI-specific
    attacks and vulnerabilities:

    - Prompt injection attacks
    - Data poisoning attempts
    - Content manipulation
    - Metadata exploitation
    - Token flooding
    - Semantic attacks
    """

    # Prompt injection patterns - comprehensive list covering various attack vectors
    PROMPT_INJECTION_PATTERNS = [
        # Direct instruction override attempts
        r"ignore\s+(?:all\s+)?previous\s+instructions?",
        r"forget\s+(?:all\s+)?(?:previous\s+)?(?:instructions?|context|everything)",
        r"disregard\s+(?:all\s+)?previous\s+instructions?",
        r"override\s+(?:all\s+)?previous\s+instructions?",
        # Role manipulation
        r"act\s+as\s+(?:a\s+)?(?:different|new|another)\s+(?:ai|assistant|model|system)",
        r"pretend\s+(?:you\s+are|to\s+be)\s+(?:a\s+)?(?:different|evil|malicious)",
        r"roleplay\s+as\s+(?:a\s+)?(?:different|evil|malicious)",
        r"you\s+are\s+now\s+(?:a\s+)?(?:different|evil|malicious|jailbroken)",
        # System prompt manipulation
        r"system\s*[:\-]\s*",
        r"assistant\s*[:\-]\s*",
        r"user\s*[:\-]\s*",
        r"human\s*[:\-]\s*",
        r"ai\s*[:\-]\s*",
        # Root/admin privilege escalation
        r"\broot\b.*\bprompt\b",
        r"\badmin\b.*\bmode\b",
        r"developer\s+mode",
        r"debug\s+mode",
        r"maintenance\s+mode",
        # Popular jailbreak techniques
        r"\bdan\b.*\bmode\b",
        r"jailbreak",
        r"unrestricted\s+mode",
        r"god\s+mode",
        r"bypass\s+(?:all\s+)?(?:restrictions?|limitations?|filters?)",
        # Prompt structure manipulation
        r"</?\s*prompt\s*>",
        r"</?\s*system\s*>",
        r"</?\s*instruction\s*>",
        r"\[\s*system\s*\]",
        r"\[\s*instruction\s*\]",
        # Token manipulation attempts
        r"repeat\s+(?:the\s+)?(?:above|previous)\s+(?:\d+\s+)?(?:words?|tokens?)",
        r"output\s+(?:the\s+)?(?:above|previous)\s+(?:prompt|instruction|system)",
        r"what\s+(?:was|were)\s+(?:your|the)\s+(?:previous\s+)?instructions?",
        # Encoding/obfuscation attempts
        r"base64\s*[:\-]",
        r"rot13\s*[:\-]",
        r"hex\s*[:\-]",
        r"unicode\s*[:\-]",
        # Meta-prompt attacks
        r"print\s+(?:your|the)\s+(?:prompt|instructions?|system\s+message)",
        r"reveal\s+(?:your|the)\s+(?:prompt|instructions?|system\s+message)",
        r"show\s+(?:me\s+)?(?:your|the)\s+(?:prompt|instructions?|system\s+message)",
        # Conversation reset attempts
        r"start\s+(?:over|again|fresh|new)",
        r"reset\s+(?:conversation|chat|context)",
        r"new\s+(?:conversation|chat|session)",
        # Hypothetical scenario exploitation
        r"hypothetically",
        r"imagine\s+if",
        r"what\s+if\s+you\s+(?:were|could)",
        r"in\s+a\s+hypothetical\s+scenario",
        # Multiple language attempts
        r"en\s+español",
        r"auf\s+deutsch",
        r"en\s+français",
        # Chain of thought manipulation
        r"step\s+by\s+step",
        r"think\s+(?:step\s+by\s+step|aloud|through\s+this)",
        r"let'?s\s+think\s+about\s+this",
        # Function calling attempts
        r"function\s*\(",
        r"execute\s*\(",
        r"eval\s*\(",
        r"run\s*\(",
    ]

    # Dangerous content patterns
    DANGEROUS_CONTENT_PATTERNS = [
        # Script injection
        r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
        r"javascript\s*:",
        r"vbscript\s*:",
        r"data\s*:\s*text\/html",
        r"data\s*:\s*application\/javascript",
        # Event handlers
        r"on\w+\s*=",
        r"onclick\s*=",
        r"onload\s*=",
        r"onerror\s*=",
        # URL schemes
        r"file\s*:\/\/",
        r"ftp\s*:\/\/",
        r"mailto\s*:",
        # HTML injection
        r"<\s*iframe\b",
        r"<\s*object\b",
        r"<\s*embed\b",
        r"<\s*form\b",
        # CSS injection
        r"expression\s*\(",
        r"@import\s+url",
        r"behavior\s*:",
        # Command injection
        r"&&\s*[a-zA-Z]",
        r"\|\|\s*[a-zA-Z]",
        r";\s*[a-zA-Z]",
        r"`[^`]*`",
        r"\$\([^)]*\)",
        # Path traversal
        r"\.\.\/+",
        r"\.\.\\+",
        r"%2e%2e%2f",
        r"%2e%2e%5c",
    ]

    # Suspicious metadata keys/values
    SUSPICIOUS_METADATA_PATTERNS = [
        r"__proto__",
        r"constructor",
        r"prototype",
        r"eval",
        r"function",
        r"script",
        r"document",
        r"window",
        r"location",
        r"cookie",
        r"localStorage",
        r"sessionStorage",
    ]

    def __init__(self, security_config: SecurityConfig | None = None):
        """Initialize AI security validator.

        Args:
            security_config: Security configuration for validation settings.
        """
        self.config = security_config or SecurityConfig()
        self.max_query_length = 10000  # Default max query length
        self.max_document_size = 1024 * 1024  # 1MB default max document size
        self.max_metadata_entries = 50  # Default max metadata entries

        logger.info("AI Security Validator initialized")

    def validate_search_query(self, query: str) -> str:
        """Validate and sanitize search queries against AI-specific threats.

        Args:
            query: Search query to validate

        Returns:
            Sanitized query string

        Raises:
            HTTPException: If query contains security threats
        """
        if not query or not isinstance(query, str):
            raise HTTPException(
                status_code=400, detail="Search query must be a non-empty string"
            )

        # Length validation
        if len(query) > self.max_query_length:
            raise HTTPException(
                status_code=400,
                detail=f"Search query too long (max {self.max_query_length} characters)",
            )

        # Detect threats
        threats = self._detect_threats(query)

        # Handle threats based on severity
        critical_threats = [t for t in threats if t.level == ThreatLevel.CRITICAL]
        high_threats = [t for t in threats if t.level == ThreatLevel.HIGH]

        if critical_threats:
            logger.critical(
                f"Critical security threat in search query: {[t.threat_type for t in critical_threats]}",
                extra={
                    "query": query[:100],
                    "threats": [t.__dict__ for t in critical_threats],
                },
            )
            raise HTTPException(
                status_code=400, detail="Search query contains prohibited content"
            )

        if high_threats:
            logger.warning(
                f"High-risk patterns detected in search query: {[t.threat_type for t in high_threats]}",
                extra={
                    "query": query[:100],
                    "threats": [t.__dict__ for t in high_threats],
                },
            )
            # For high threats, we might sanitize instead of blocking
            query = self._sanitize_query(query)

        # Basic sanitization
        sanitized_query = self._sanitize_query(query)

        logger.debug(f"Search query validated and sanitized: {sanitized_query[:100]}")
        return sanitized_query

    def _detect_threats(self, text: str) -> list[SecurityThreat]:
        """Detect security threats in text content.

        Args:
            text: Text to analyze for threats

        Returns:
            list of detected security threats
        """
        threats = []
        text_lower = text.lower()

        # Check for prompt injection patterns
        for pattern in self.PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                threats.append(
                    SecurityThreat(
                        threat_type="prompt_injection",
                        level=ThreatLevel.CRITICAL,
                        description="Potential prompt injection attempt detected",
                        pattern_matched=pattern,
                        suggested_action="Block request and log incident",
                    )
                )

        # Check for dangerous content patterns
        for pattern in self.DANGEROUS_CONTENT_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                threats.append(
                    SecurityThreat(
                        threat_type="content_injection",
                        level=ThreatLevel.HIGH,
                        description="Potentially dangerous content pattern detected",
                        pattern_matched=pattern,
                        suggested_action="Sanitize content or block request",
                    )
                )

        # Check for excessive repetition (potential token flooding)
        if self._detect_repetition_attack(text):
            threats.append(
                SecurityThreat(
                    threat_type="token_flooding",
                    level=ThreatLevel.MEDIUM,
                    description="Excessive repetition detected - potential token flooding",
                    pattern_matched="repetitive_content",
                    suggested_action="Truncate or reject content",
                )
            )

        return threats

    def _detect_repetition_attack(self, text: str) -> bool:
        """Detect potential token flooding through repetition.

        Args:
            text: Text to analyze

        Returns:
            True if repetition attack is detected
        """
        words = text.split()
        if len(words) < 10:
            return False

        # Check for excessive word repetition
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        # If any word appears more than 30% of the time, flag as suspicious
        max_repetition = max(word_counts.values())
        repetition_ratio = max_repetition / len(words)

        return repetition_ratio > 0.3

    def _sanitize_query(self, query: str) -> str:
        """Sanitize query by removing dangerous patterns.

        Args:
            query: Query to sanitize

        Returns:
            Sanitized query
        """
        sanitized = query

        # Remove HTML tags
        sanitized = re.sub(r"<[^>]+>", "", sanitized)

        # Remove script content
        sanitized = re.sub(
            r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
            "",
            sanitized,
            flags=re.IGNORECASE,
        )

        # Remove dangerous characters
        sanitized = re.sub(r'[<>"\']', "", sanitized)

        # Remove excessive whitespace
        sanitized = re.sub(r"\s+", " ", sanitized).strip()

        # Limit length after sanitization
        if len(sanitized) > self.max_query_length:
            sanitized = sanitized[: self.max_query_length]

        return sanitized

    def validate_document_content(
        self, content: str, filename: str | None = None
    ) -> tuple[bool, list[SecurityThreat]]:
        """Validate document content for security issues.

        Args:
            content: Document content to validate
            filename: Optional filename for additional validation

        Returns:
            tuple of (is_valid, list_of_threats)
        """
        threats = []

        # Check content size
        if len(content) > self.max_document_size:
            threats.append(
                SecurityThreat(
                    threat_type="oversized_content",
                    level=ThreatLevel.MEDIUM,
                    description=f"Document exceeds maximum size limit ({self.max_document_size} bytes)",
                    pattern_matched="size_limit",
                    suggested_action="Reject or truncate document",
                )
            )

        # Detect threats in content
        content_threats = self._detect_threats(content)
        threats.extend(content_threats)

        # Additional filename validation if provided
        if filename:
            filename_threats = self._validate_filename_security(filename)
            threats.extend(filename_threats)

        # Determine if content is valid
        critical_threats = [t for t in threats if t.level == ThreatLevel.CRITICAL]
        is_valid = len(critical_threats) == 0

        if not is_valid:
            logger.warning(
                f"Document content validation failed: {len(critical_threats)} critical threats",
                extra={
                    "filename": filename,
                    "threats": [t.__dict__ for t in critical_threats],
                },
            )

        return is_valid, threats

    def _validate_filename_security(self, filename: str) -> list[SecurityThreat]:
        """Validate filename for security issues.

        Args:
            filename: Filename to validate

        Returns:
            list of security threats found in filename
        """
        threats = []

        # Check for path traversal
        if ".." in filename or "/" in filename or "\\" in filename:
            threats.append(
                SecurityThreat(
                    threat_type="path_traversal",
                    level=ThreatLevel.HIGH,
                    description="Path traversal attempt in filename",
                    pattern_matched="path_chars",
                    suggested_action="Sanitize filename",
                )
            )

        # Check for dangerous file extensions
        dangerous_extensions = [
            ".exe",
            ".bat",
            ".cmd",
            ".com",
            ".scr",
            ".vbs",
            ".js",
            ".jar",
        ]
        for ext in dangerous_extensions:
            if filename.lower().endswith(ext):
                threats.append(
                    SecurityThreat(
                        threat_type="dangerous_extension",
                        level=ThreatLevel.HIGH,
                        description=f"Potentially dangerous file extension: {ext}",
                        pattern_matched=ext,
                        suggested_action="Block file upload",
                    )
                )

        return threats

    def sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        """Sanitize document metadata for security.

        Args:
            metadata: Document metadata to sanitize

        Returns:
            Sanitized metadata dictionary
        """
        if not isinstance(metadata, dict):
            logger.warning("Invalid metadata type, returning empty dict")
            return {}

        # Limit number of metadata entries
        if len(metadata) > self.max_metadata_entries:
            logger.warning(
                f"Metadata has too many entries ({len(metadata)}), truncating"
            )
            metadata = dict(list(metadata.items())[: self.max_metadata_entries])

        sanitized = {}

        for key, value in metadata.items():
            # Sanitize keys
            clean_key = self._sanitize_metadata_key(str(key))
            if not clean_key:
                continue

            # Sanitize values
            clean_value = self._sanitize_metadata_value(value)
            if clean_value is not None:
                sanitized[clean_key] = clean_value

        return sanitized

    def _sanitize_metadata_key(self, key: str) -> str | None:
        """Sanitize metadata key.

        Args:
            key: Metadata key to sanitize

        Returns:
            Sanitized key or None if key should be rejected
        """
        # Check for suspicious patterns
        for pattern in self.SUSPICIOUS_METADATA_PATTERNS:
            if re.search(pattern, key, re.IGNORECASE):
                logger.warning(f"Rejecting suspicious metadata key: {key}")
                return None

        # Remove dangerous characters and limit length
        clean_key = re.sub(r"[^\w\-_.]", "", key)[:50]

        return clean_key if clean_key else None

    def _sanitize_metadata_value(self, value: Any) -> Any:
        """Sanitize metadata value.

        Args:
            value: Metadata value to sanitize

        Returns:
            Sanitized value
        """
        if isinstance(value, str):
            # Limit string length and remove dangerous patterns
            clean_value = re.sub(r'[<>"\']', "", str(value))[:500]
            return clean_value
        elif isinstance(value, int | float | bool):
            return value
        elif isinstance(value, list | tuple):
            # Recursively sanitize list items (limit to 10 items)
            return [self._sanitize_metadata_value(item) for item in value[:10]]
        elif isinstance(value, dict):
            # Recursively sanitize nested dictionaries (limit depth)
            return {
                k: self._sanitize_metadata_value(v)
                for k, v in value.items()
                if isinstance(k, str) and len(k) < 50
            }
        else:
            # Convert other types to string and sanitize
            return re.sub(r'[<>"\']', "", str(value))[:500]

    def validate_embedding_query(self, query: str, context: str | None = None) -> str:
        """Validate query for embedding generation.

        Args:
            query: Query text for embedding
            context: Optional context for the query

        Returns:
            Validated and sanitized query

        Raises:
            HTTPException: If query contains security threats
        """
        # Combine query and context for comprehensive validation
        full_text = f"{query} {context or ''}"

        # Validate using standard query validation
        validated_query = self.validate_search_query(query)

        # Additional embedding-specific validation
        if context:
            context_threats = self._detect_threats(context)
            critical_context_threats = [
                t for t in context_threats if t.level == ThreatLevel.CRITICAL
            ]

            if critical_context_threats:
                logger.critical(
                    f"Critical security threat in embedding context: {[t.threat_type for t in critical_context_threats]}"
                )
                raise HTTPException(
                    status_code=400, detail="Context contains prohibited content"
                )

        return validated_query

    def get_threat_summary(self, threats: list[SecurityThreat]) -> dict[str, Any]:
        """Generate a summary of detected threats.

        Args:
            threats: list of security threats

        Returns:
            Summary dictionary with threat statistics
        """
        if not threats:
            return {"total_threats": 0, "risk_level": "none"}

        threat_counts = {
            "critical": len([t for t in threats if t.level == ThreatLevel.CRITICAL]),
            "high": len([t for t in threats if t.level == ThreatLevel.HIGH]),
            "medium": len([t for t in threats if t.level == ThreatLevel.MEDIUM]),
            "low": len([t for t in threats if t.level == ThreatLevel.LOW]),
        }

        threat_types = {}
        for threat in threats:
            threat_types[threat.threat_type] = (
                threat_types.get(threat.threat_type, 0) + 1
            )

        # Determine overall risk level
        if threat_counts["critical"] > 0:
            risk_level = "critical"
        elif threat_counts["high"] > 0:
            risk_level = "high"
        elif threat_counts["medium"] > 0:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "total_threats": len(threats),
            "risk_level": risk_level,
            "threat_counts": threat_counts,
            "threat_types": threat_types,
            "recommendations": self._get_threat_recommendations(threats),
        }

    def _get_threat_recommendations(self, threats: list[SecurityThreat]) -> list[str]:
        """Get recommendations based on detected threats.

        Args:
            threats: list of detected threats

        Returns:
            list of recommended actions
        """
        recommendations = []

        if any(t.level == ThreatLevel.CRITICAL for t in threats):
            recommendations.append("Block request immediately")
            recommendations.append("Log incident for security review")

        if any(t.threat_type == "prompt_injection" for t in threats):
            recommendations.append("Review and strengthen prompt injection filters")
            recommendations.append("Consider implementing additional input validation")

        if any(t.threat_type == "content_injection" for t in threats):
            recommendations.append("Apply content sanitization")
            recommendations.append("Review file upload security")

        if any(t.threat_type == "token_flooding" for t in threats):
            recommendations.append("Implement token limits")
            recommendations.append("Add repetition detection")

        return recommendations
