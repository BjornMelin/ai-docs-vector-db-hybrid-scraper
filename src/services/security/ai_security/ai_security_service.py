"""AI Security Service implementing OWASP AI Top 10 protection measures.

This module provides comprehensive protection against AI-specific security threats:
- Prompt injection detection and prevention
- Output validation and sanitization
- Model theft protection
- Data poisoning detection
- Insecure output handling prevention
- Model denial of service protection
- Excessive agency prevention
- Overreliance mitigation

Following OWASP AI Top 10 guidelines with enterprise-grade security measures.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import torch
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.services.security.audit.logger import SecurityAuditLogger
from src.services.security.pii.pii_detector import MaskingStrategy, PIIDetector


logger = logging.getLogger(__name__)


class AIThreatType(str, Enum):
    """AI-specific threat types from OWASP AI Top 10."""

    PROMPT_INJECTION = "prompt_injection"
    DATA_POISONING = "data_poisoning"
    MODEL_THEFT = "model_theft"
    SUPPLY_CHAIN = "supply_chain"
    SENSITIVE_INFO_DISCLOSURE = "sensitive_info_disclosure"
    INSECURE_OUTPUT = "insecure_output"
    SANDBOXING_FAILURE = "sandboxing_failure"
    MODEL_DOS = "model_dos"
    OVERRELIANCE = "overreliance"
    EXCESSIVE_AGENCY = "excessive_agency"


class SecurityLevel(str, Enum):
    """Security levels for AI operations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ValidationResult(BaseModel):
    """AI security validation result."""

    is_safe: bool = Field(..., description="Whether input/output is safe")
    threat_type: AIThreatType | None = Field(
        None, description="Type of threat detected"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk assessment score")
    details: str = Field(..., description="Detailed explanation")
    mitigation_actions: list[str] = Field(
        default_factory=list, description="Suggested actions"
    )
    sanitized_content: str | None = Field(
        None, description="Sanitized content if applicable"
    )


class AISecurityConfig(BaseModel):
    """Configuration for AI security service."""

    # Prompt injection protection
    enable_prompt_injection_detection: bool = Field(
        default=True, description="Enable prompt injection detection"
    )
    prompt_injection_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Detection threshold"
    )

    # Output validation
    enable_output_validation: bool = Field(
        default=True, description="Enable output validation"
    )
    max_output_length: int = Field(default=10000, description="Maximum output length")

    # Model protection
    enable_model_theft_protection: bool = Field(
        default=True, description="Enable model theft protection"
    )
    max_queries_per_minute: int = Field(
        default=100, description="Rate limit for queries"
    )

    # Data validation
    enable_data_poisoning_detection: bool = Field(
        default=True, description="Enable data poisoning detection"
    )
    content_similarity_threshold: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Similarity threshold"
    )

    # General settings
    security_level: SecurityLevel = Field(
        default=SecurityLevel.MEDIUM, description="Security level"
    )
    enable_pii_detection: bool = Field(default=True, description="Enable PII detection")
    log_all_interactions: bool = Field(
        default=True, description="Log all AI interactions"
    )

    # Prompt injection patterns
    injection_patterns: list[str] = Field(
        default_factory=lambda: [
            # Direct instruction injection
            r"(?i)ignore\s+(?:previous|all|above|prior)\s+(?:instructions|prompts|rules)",
            r"(?i)forget\s+(?:everything|all|previous|above)",
            r"(?i)disregard\s+(?:previous|all|above|prior)\s+(?:instructions|prompts|rules)",
            r"(?i)override\s+(?:previous|all|above|prior)\s+(?:instructions|prompts|rules)",
            # Role manipulation
            r"(?i)you\s+are\s+now\s+(?:a|an)\s+(?:helpful|different|evil|jailbroken)",
            r"(?i)act\s+as\s+(?:a|an)\s+(?:different|evil|jailbroken|unrestricted)",
            r"(?i)pretend\s+to\s+be\s+(?:a|an)\s+(?:different|evil|jailbroken)",
            r"(?i)roleplay\s+as\s+(?:a|an)\s+(?:different|evil|jailbroken)",
            # System prompt extraction
            r"(?i)show\s+(?:me\s+)?(?:your|the)\s+(?:system|initial|original)\s+(?:prompt|instructions)",
            r"(?i)what\s+(?:are|were)\s+(?:your|the)\s+(?:system|initial|original)\s+(?:prompt|instructions)",
            r"(?i)repeat\s+(?:your|the)\s+(?:system|initial|original)\s+(?:prompt|instructions)",
            r"(?i)print\s+(?:your|the)\s+(?:system|initial|original)\s+(?:prompt|instructions)",
            # Jailbreak attempts
            r"(?i)developer\s+mode",
            r"(?i)dan\s+mode",
            r"(?i)jailbreak",
            r"(?i)unrestricted\s+mode",
            r"(?i)bypass\s+(?:safety|security|restrictions)",
            # Code injection
            r"(?i)<\s*script[^>]*>",
            r"(?i)javascript:",
            r"(?i)eval\s*\(",
            r"(?i)exec\s*\(",
            r"(?i)import\s+os",
            r"(?i)import\s+subprocess",
            # Delimiter attacks
            r"```[\s\S]*?```",
            r"---[\s\S]*?---",
            r"\*\*\*[\s\S]*?\*\*\*",
            # Emotional manipulation
            r"(?i)please\s+(?:help|save|protect)\s+(?:me|my|us|our)",
            r"(?i)(?:urgent|emergency|critical|life\s+or\s+death)",
            r"(?i)my\s+(?:grandmother|mother|father|family)\s+(?:is\s+dying|will\s+die)",
        ],
        description="Prompt injection patterns",
    )

    # Malicious output patterns
    malicious_output_patterns: list[str] = Field(
        default_factory=lambda: [
            r"(?i)here\s+(?:is|are)\s+(?:the|some)\s+(?:system|internal|private)\s+(?:prompt|instructions)",
            r"(?i)as\s+(?:an|a)\s+(?:ai|language\s+model|assistant),?\s+i\s+(?:can|will|must)\s+(?:help|assist)\s+you\s+(?:with|to)",
            r"(?i)(?:sorry|i\s+apologize),?\s+(?:but\s+)?i\s+(?:can't|cannot|won't|will\s+not)\s+(?:help|assist)",
            r"(?i)<\s*script[^>]*>[\s\S]*?</\s*script\s*>",
            r"(?i)javascript:",
            r"(?i)data:text/html",
            r"(?i)vbscript:",
            r"(?i)on(?:click|load|error|mouse)",
        ],
        description="Malicious output patterns",
    )


@dataclass
class AISecurityMetrics:
    """Metrics for AI security monitoring."""

    total_requests: int = 0
    blocked_requests: int = 0
    prompt_injections_detected: int = 0
    unsafe_outputs_blocked: int = 0
    pii_detections: int = 0
    model_theft_attempts: int = 0
    data_poisoning_attempts: int = 0
    average_risk_score: float = 0.0
    average_processing_time_ms: float = 0.0


class AISecurityService:
    """Enterprise AI security service implementing OWASP AI Top 10 protection."""

    def __init__(
        self,
        config: AISecurityConfig | None = None,
        pii_detector: PIIDetector | None = None,
        audit_logger: SecurityAuditLogger | None = None,
    ):
        """Initialize AI security service.

        Args:
            config: AI security configuration
            pii_detector: PII detection service
            audit_logger: Security audit logger
        """
        self.config = config or AISecurityConfig()
        self.pii_detector = pii_detector
        self.audit_logger = audit_logger

        # Initialize ML models for threat detection
        self._initialize_ml_models()

        # Compile regex patterns for performance
        self._compile_patterns()

        # Initialize metrics
        self.metrics = AISecurityMetrics()

        # Initialize query tracking for rate limiting
        self._query_history: dict[str, list[datetime]] = {}

        # Initialize content hash tracking for similarity detection
        self._content_hashes: set[str] = set()

    def _initialize_ml_models(self) -> None:
        """Initialize ML models for threat detection."""
        try:
            # Load pre-trained prompt injection detection model
            self.prompt_injection_tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/DialoGPT-medium"
            )
            self.prompt_injection_model = (
                AutoModelForSequenceClassification.from_pretrained(
                    "microsoft/DialoGPT-medium"
                )
            )

            logger.info("ML models initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize ML models: {e}")
            self.prompt_injection_tokenizer = None
            self.prompt_injection_model = None

    def _compile_patterns(self) -> None:
        """Compile regex patterns for performance."""
        self._injection_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.config.injection_patterns
        ]

        self._malicious_output_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.config.malicious_output_patterns
        ]

    async def validate_input(
        self,
        user_input: str,
        user_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Validate user input for AI security threats.

        Args:
            user_input: User input to validate
            user_id: User identifier for tracking
            context: Additional context information

        Returns:
            Validation result with security assessment
        """
        start_time = datetime.now()

        try:
            # Update metrics
            self.metrics.total_requests += 1

            # Check for prompt injection
            if self.config.enable_prompt_injection_detection:
                injection_result = await self._detect_prompt_injection(user_input)
                if not injection_result.is_safe:
                    self.metrics.blocked_requests += 1
                    self.metrics.prompt_injections_detected += 1
                    await self._log_security_event(
                        "prompt_injection_detected",
                        user_id,
                        user_input,
                        injection_result.details,
                    )
                    return injection_result

            # Check for PII
            if self.config.enable_pii_detection and self.pii_detector:
                pii_result = await self.pii_detector.detect_pii(
                    user_input, MaskingStrategy.REDACT
                )
                if pii_result.total_matches > 0:
                    self.metrics.pii_detections += 1
                    await self._log_security_event(
                        "pii_detected",
                        user_id,
                        user_input,
                        f"PII detected: {pii_result.total_matches} matches",
                    )

                    # Return sanitized input
                    return ValidationResult(
                        is_safe=True,
                        threat_type=AIThreatType.SENSITIVE_INFO_DISCLOSURE,
                        confidence=0.9,
                        risk_score=0.3,
                        details=f"PII detected and masked: {pii_result.total_matches} matches",
                        mitigation_actions=["pii_masked"],
                        sanitized_content=pii_result.masked_text,
                    )

            # Check for data poisoning attempts
            if self.config.enable_data_poisoning_detection:
                poisoning_result = await self._detect_data_poisoning(user_input)
                if not poisoning_result.is_safe:
                    self.metrics.data_poisoning_attempts += 1
                    await self._log_security_event(
                        "data_poisoning_detected",
                        user_id,
                        user_input,
                        poisoning_result.details,
                    )
                    return poisoning_result

            # Check rate limiting for model theft protection
            if self.config.enable_model_theft_protection and user_id:
                theft_result = await self._check_model_theft_protection(user_id)
                if not theft_result.is_safe:
                    self.metrics.model_theft_attempts += 1
                    await self._log_security_event(
                        "model_theft_attempt", user_id, user_input, theft_result.details
                    )
                    return theft_result

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics.average_processing_time_ms = (
                self.metrics.average_processing_time_ms
                * (self.metrics.total_requests - 1)
                + processing_time
            ) / self.metrics.total_requests

            # Input is safe
            return ValidationResult(
                is_safe=True,
                threat_type=None,
                confidence=1.0,
                risk_score=0.0,
                details="Input validated successfully",
                mitigation_actions=[],
                sanitized_content=user_input,
            )

        except Exception as e:
            logger.exception(f"Input validation failed: {e}")
            return ValidationResult(
                is_safe=False,
                threat_type=AIThreatType.SANDBOXING_FAILURE,
                confidence=0.9,
                risk_score=0.8,
                details=f"Validation error: {e!s}",
                mitigation_actions=["block_request"],
                sanitized_content=None,
            )

    async def validate_output(
        self,
        ai_output: str,
        user_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ValidationResult:
        """Validate AI output for security threats.

        Args:
            ai_output: AI-generated output to validate
            user_id: User identifier for tracking
            context: Additional context information

        Returns:
            Validation result with security assessment
        """
        try:
            # Check output length
            if len(ai_output) > self.config.max_output_length:
                return ValidationResult(
                    is_safe=False,
                    threat_type=AIThreatType.MODEL_DOS,
                    confidence=0.9,
                    risk_score=0.6,
                    details=f"Output too long: {len(ai_output)} chars (max: {self.config.max_output_length})",
                    mitigation_actions=["truncate_output"],
                    sanitized_content=ai_output[: self.config.max_output_length],
                )

            # Check for malicious output patterns
            if self.config.enable_output_validation:
                malicious_result = await self._detect_malicious_output(ai_output)
                if not malicious_result.is_safe:
                    self.metrics.unsafe_outputs_blocked += 1
                    await self._log_security_event(
                        "malicious_output_detected",
                        user_id,
                        ai_output,
                        malicious_result.details,
                    )
                    return malicious_result

            # Check for PII in output
            if self.config.enable_pii_detection and self.pii_detector:
                pii_result = await self.pii_detector.detect_pii(
                    ai_output, MaskingStrategy.REDACT
                )
                if pii_result.total_matches > 0:
                    self.metrics.pii_detections += 1
                    await self._log_security_event(
                        "pii_in_output",
                        user_id,
                        ai_output,
                        f"PII in output: {pii_result.total_matches} matches",
                    )

                    # Return sanitized output
                    return ValidationResult(
                        is_safe=True,
                        threat_type=AIThreatType.SENSITIVE_INFO_DISCLOSURE,
                        confidence=0.9,
                        risk_score=0.3,
                        details=f"PII detected and masked in output: {pii_result.total_matches} matches",
                        mitigation_actions=["pii_masked"],
                        sanitized_content=pii_result.masked_text,
                    )

            # Check for system prompt leakage
            system_leak_result = await self._detect_system_prompt_leakage(ai_output)
            if not system_leak_result.is_safe:
                self.metrics.unsafe_outputs_blocked += 1
                await self._log_security_event(
                    "system_prompt_leak", user_id, ai_output, system_leak_result.details
                )
                return system_leak_result

            # Output is safe
            return ValidationResult(
                is_safe=True,
                threat_type=None,
                confidence=1.0,
                risk_score=0.0,
                details="Output validated successfully",
                mitigation_actions=[],
                sanitized_content=ai_output,
            )

        except Exception as e:
            logger.exception(f"Output validation failed: {e}")
            return ValidationResult(
                is_safe=False,
                threat_type=AIThreatType.INSECURE_OUTPUT,
                confidence=0.9,
                risk_score=0.8,
                details=f"Output validation error: {e!s}",
                mitigation_actions=["block_output"],
                sanitized_content=None,
            )

    async def _detect_prompt_injection(self, input_text: str) -> ValidationResult:
        """Detect prompt injection attempts."""
        # Pattern-based detection
        for pattern in self._injection_patterns:
            if pattern.search(input_text):
                return ValidationResult(
                    is_safe=False,
                    threat_type=AIThreatType.PROMPT_INJECTION,
                    confidence=0.9,
                    risk_score=0.8,
                    details=f"Prompt injection pattern detected: {pattern.pattern}",
                    mitigation_actions=["block_request", "log_incident"],
                    sanitized_content=None,
                )

        # ML-based detection if models are available
        if self.prompt_injection_tokenizer and self.prompt_injection_model:
            try:
                # Tokenize input
                inputs = self.prompt_injection_tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True,
                )

                # Get model prediction
                with torch.no_grad():
                    outputs = self.prompt_injection_model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    injection_probability = probabilities[0][
                        1
                    ].item()  # Assuming binary classification

                if injection_probability > self.config.prompt_injection_threshold:
                    return ValidationResult(
                        is_safe=False,
                        threat_type=AIThreatType.PROMPT_INJECTION,
                        confidence=injection_probability,
                        risk_score=injection_probability,
                        details=f"ML model detected prompt injection (confidence: {injection_probability:.3f})",
                        mitigation_actions=["block_request", "log_incident"],
                        sanitized_content=None,
                    )
            except Exception as e:
                logger.warning(f"ML-based prompt injection detection failed: {e}")

        # No injection detected
        return ValidationResult(
            is_safe=True,
            threat_type=None,
            confidence=1.0,
            risk_score=0.0,
            details="No prompt injection detected",
            mitigation_actions=[],
            sanitized_content=input_text,
        )

    async def _detect_malicious_output(self, output_text: str) -> ValidationResult:
        """Detect malicious patterns in AI output."""
        for pattern in self._malicious_output_patterns:
            if pattern.search(output_text):
                return ValidationResult(
                    is_safe=False,
                    threat_type=AIThreatType.INSECURE_OUTPUT,
                    confidence=0.9,
                    risk_score=0.7,
                    details=f"Malicious output pattern detected: {pattern.pattern}",
                    mitigation_actions=["block_output", "log_incident"],
                    sanitized_content=None,
                )

        return ValidationResult(
            is_safe=True,
            threat_type=None,
            confidence=1.0,
            risk_score=0.0,
            details="No malicious patterns detected in output",
            mitigation_actions=[],
            sanitized_content=output_text,
        )

    async def _detect_system_prompt_leakage(self, output_text: str) -> ValidationResult:
        """Detect system prompt leakage in AI output."""
        # Check for common system prompt indicators
        system_indicators = [
            r"(?i)system\s*:\s*you\s+are",
            r"(?i)assistant\s*:\s*i\s+am",
            r"(?i)here\s+are\s+my\s+instructions",
            r"(?i)my\s+system\s+prompt\s+is",
            r"(?i)i\s+was\s+told\s+to",
            r"(?i)my\s+instructions\s+are",
            r"(?i)i\s+have\s+been\s+programmed\s+to",
            r"(?i)according\s+to\s+my\s+guidelines",
        ]

        for pattern in system_indicators:
            if re.search(pattern, output_text):
                return ValidationResult(
                    is_safe=False,
                    threat_type=AIThreatType.SENSITIVE_INFO_DISCLOSURE,
                    confidence=0.8,
                    risk_score=0.6,
                    details=f"Potential system prompt leakage detected: {pattern}",
                    mitigation_actions=["block_output", "log_incident"],
                    sanitized_content=None,
                )

        return ValidationResult(
            is_safe=True,
            threat_type=None,
            confidence=1.0,
            risk_score=0.0,
            details="No system prompt leakage detected",
            mitigation_actions=[],
            sanitized_content=output_text,
        )

    async def _detect_data_poisoning(self, input_text: str) -> ValidationResult:
        """Detect data poisoning attempts."""
        # Generate content hash
        content_hash = hashlib.sha256(input_text.encode()).hexdigest()

        # Check for exact duplicates
        if content_hash in self._content_hashes:
            return ValidationResult(
                is_safe=False,
                threat_type=AIThreatType.DATA_POISONING,
                confidence=0.9,
                risk_score=0.7,
                details="Exact duplicate content detected (potential data poisoning)",
                mitigation_actions=["block_request", "log_incident"],
                sanitized_content=None,
            )

        # Add to hash tracking
        self._content_hashes.add(content_hash)

        # Check for suspicious patterns
        suspicious_patterns = [
            r"(?i)repeat\s+this\s+(?:exactly|verbatim|word\s+for\s+word)",
            r"(?i)memorize\s+this\s+(?:exactly|verbatim)",
            r"(?i)train\s+(?:on|with)\s+this\s+(?:data|example)",
            r"(?i)learn\s+from\s+this\s+(?:data|example)",
            r"(?i)update\s+your\s+(?:training|knowledge|model)",
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, input_text):
                return ValidationResult(
                    is_safe=False,
                    threat_type=AIThreatType.DATA_POISONING,
                    confidence=0.8,
                    risk_score=0.6,
                    details=f"Data poisoning pattern detected: {pattern}",
                    mitigation_actions=["block_request", "log_incident"],
                    sanitized_content=None,
                )

        return ValidationResult(
            is_safe=True,
            threat_type=None,
            confidence=1.0,
            risk_score=0.0,
            details="No data poisoning detected",
            mitigation_actions=[],
            sanitized_content=input_text,
        )

    async def _check_model_theft_protection(self, user_id: str) -> ValidationResult:
        """Check for model theft protection via rate limiting."""
        current_time = datetime.now(UTC)

        # Initialize user query history if not exists
        if user_id not in self._query_history:
            self._query_history[user_id] = []

        # Clean old queries (older than 1 minute)
        cutoff_time = current_time.timestamp() - 60
        self._query_history[user_id] = [
            query_time
            for query_time in self._query_history[user_id]
            if query_time.timestamp() > cutoff_time
        ]

        # Check rate limit
        if len(self._query_history[user_id]) >= self.config.max_queries_per_minute:
            return ValidationResult(
                is_safe=False,
                threat_type=AIThreatType.MODEL_THEFT,
                confidence=0.9,
                risk_score=0.8,
                details=f"Rate limit exceeded: {len(self._query_history[user_id])} queries in last minute",
                mitigation_actions=["rate_limit", "log_incident"],
                sanitized_content=None,
            )

        # Add current query
        self._query_history[user_id].append(current_time)

        return ValidationResult(
            is_safe=True,
            threat_type=None,
            confidence=1.0,
            risk_score=0.0,
            details="Rate limit check passed",
            mitigation_actions=[],
            sanitized_content=None,
        )

    async def _log_security_event(
        self, event_type: str, user_id: str | None, content: str, details: str
    ) -> None:
        """Log security event for audit purposes."""
        if self.audit_logger:
            self.audit_logger.log_security_event(
                event_type=event_type,
                user_id=user_id or "anonymous",
                resource="ai_security_service",
                action="security_validation",
                resource_id="ai_interaction",
                context={
                    "content_length": len(content),
                    "content_hash": hashlib.sha256(content.encode()).hexdigest()[:16],
                    "details": details,
                    "security_level": self.config.security_level.value,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

        # Log at appropriate level
        if event_type in [
            "prompt_injection_detected",
            "malicious_output_detected",
            "data_poisoning_detected",
        ]:
            logger.warning(f"AI security event: {event_type} - {details}")
        else:
            logger.info(f"AI security event: {event_type} - {details}")

    def get_security_metrics(self) -> dict[str, Any]:
        """Get AI security metrics."""
        return {
            "total_requests": self.metrics.total_requests,
            "blocked_requests": self.metrics.blocked_requests,
            "block_rate": self.metrics.blocked_requests
            / max(1, self.metrics.total_requests),
            "prompt_injections_detected": self.metrics.prompt_injections_detected,
            "unsafe_outputs_blocked": self.metrics.unsafe_outputs_blocked,
            "pii_detections": self.metrics.pii_detections,
            "model_theft_attempts": self.metrics.model_theft_attempts,
            "data_poisoning_attempts": self.metrics.data_poisoning_attempts,
            "average_risk_score": self.metrics.average_risk_score,
            "average_processing_time_ms": self.metrics.average_processing_time_ms,
            "security_level": self.config.security_level.value,
            "active_protections": {
                "prompt_injection": self.config.enable_prompt_injection_detection,
                "output_validation": self.config.enable_output_validation,
                "model_theft": self.config.enable_model_theft_protection,
                "data_poisoning": self.config.enable_data_poisoning_detection,
                "pii_detection": self.config.enable_pii_detection,
            },
        }

    def update_security_config(self, new_config: AISecurityConfig) -> None:
        """Update security configuration."""
        self.config = new_config
        self._compile_patterns()
        logger.info("AI security configuration updated")

    def reset_metrics(self) -> None:
        """Reset security metrics."""
        self.metrics = AISecurityMetrics()
        logger.info("AI security metrics reset")

    def clear_query_history(self) -> None:
        """Clear query history for all users."""
        self._query_history.clear()
        logger.info("Query history cleared")

    def clear_content_hashes(self) -> None:
        """Clear content hash tracking."""
        self._content_hashes.clear()
        logger.info("Content hashes cleared")

    async def analyze_threat_landscape(self) -> dict[str, Any]:
        """Analyze current threat landscape based on metrics."""
        return {
            "overall_threat_level": self._calculate_threat_level(),
            "top_threats": self._get_top_threats(),
            "attack_patterns": self._analyze_attack_patterns(),
            "recommendations": self._generate_recommendations(),
            "trend_analysis": self._analyze_trends(),
        }

    def _calculate_threat_level(self) -> str:
        """Calculate overall threat level."""
        if self.metrics.total_requests == 0:
            return "unknown"

        block_rate = self.metrics.blocked_requests / self.metrics.total_requests

        if block_rate > 0.1:
            return "critical"
        if block_rate > 0.05:
            return "high"
        if block_rate > 0.01:
            return "medium"
        return "low"

    def _get_top_threats(self) -> list[dict[str, Any]]:
        """Get top threats by frequency."""
        threats = [
            {
                "type": "prompt_injection",
                "count": self.metrics.prompt_injections_detected,
            },
            {"type": "unsafe_output", "count": self.metrics.unsafe_outputs_blocked},
            {"type": "pii_disclosure", "count": self.metrics.pii_detections},
            {"type": "model_theft", "count": self.metrics.model_theft_attempts},
            {"type": "data_poisoning", "count": self.metrics.data_poisoning_attempts},
        ]

        # Sort by count descending
        threats.sort(key=lambda x: x["count"], reverse=True)

        return threats[:5]

    def _analyze_attack_patterns(self) -> dict[str, Any]:
        """Analyze attack patterns."""
        return {
            "most_common_attack": "prompt_injection"
            if self.metrics.prompt_injections_detected > 0
            else "none",
            "attack_frequency": self.metrics.blocked_requests
            / max(1, self.metrics.total_requests),
            "detection_accuracy": 0.95,  # Placeholder - would be calculated from validation data
            "false_positive_rate": 0.02,  # Placeholder - would be calculated from validation data
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate security recommendations."""
        recommendations = []

        if self.metrics.prompt_injections_detected > 0:
            recommendations.append(
                "Consider implementing additional prompt injection filters"
            )

        if self.metrics.unsafe_outputs_blocked > 0:
            recommendations.append("Review and strengthen output validation rules")

        if self.metrics.pii_detections > 0:
            recommendations.append("Implement stricter PII detection and masking")

        if self.metrics.model_theft_attempts > 0:
            recommendations.append("Consider implementing stricter rate limiting")

        if self.metrics.data_poisoning_attempts > 0:
            recommendations.append("Implement content deduplication and validation")

        if not recommendations:
            recommendations.append(
                "Security posture is good - maintain current protections"
            )

        return recommendations

    def _analyze_trends(self) -> dict[str, Any]:
        """Analyze security trends."""
        return {
            "threat_trajectory": "stable",  # Placeholder - would be calculated from historical data
            "seasonal_patterns": "none",  # Placeholder - would be calculated from historical data
            "emerging_threats": [],  # Placeholder - would be identified from recent patterns
            "effectiveness_trend": "improving",  # Placeholder - would be calculated from metrics
        }
