"""Exception classes for the llm-grok library.

This module defines custom exceptions for handling various API errors and
validation issues.
"""

from typing import Any, Optional

__all__ = [
    "GrokError",
    "RateLimitError",
    "QuotaExceededError",
    "ValidationError",
    "ConversionError",
    "APIError",
    "AuthenticationError",
    "NetworkError",
]


class GrokError(Exception):
    """Base exception for all Grok API errors.

    Attributes:
        message: Human-readable error message
        details: Additional error details from the API
        error_code: API error code if available
    """

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
        error_code: Optional[str] = None
    ) -> None:
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        super().__init__(message)

    def __str__(self) -> str:
        """Format error message with details."""
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class RateLimitError(GrokError):
    """Exception raised when API rate limit is exceeded.

    Attributes:
        retry_after: Seconds to wait before retrying (if provided by API)
    """

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        details: Optional[dict[str, Any]] = None,
        retry_after: Optional[int] = None
    ) -> None:
        super().__init__(message, details, error_code="rate_limit_error")
        self.retry_after = retry_after


class QuotaExceededError(GrokError):
    """Exception raised when API quota is exceeded.

    This typically indicates the account has reached its usage limits.
    """

    def __init__(
        self,
        message: str = "API quota exceeded",
        details: Optional[dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details, error_code="quota_exceeded_error")


class ValidationError(GrokError):
    """Exception raised for validation failures.

    This includes invalid parameters, malformed requests, or data format issues.

    Attributes:
        field: The specific field that failed validation (if applicable)
    """

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details, error_code="validation_error")
        self.field = field


class ConversionError(GrokError):
    """Exception raised during format conversion between APIs.

    This occurs when converting between OpenAI and Anthropic formats fails.

    Attributes:
        source_format: The format being converted from
        target_format: The format being converted to
    """

    def __init__(
        self,
        message: str,
        source_format: Optional[str] = None,
        target_format: Optional[str] = None,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details, error_code="conversion_error")
        self.source_format = source_format
        self.target_format = target_format


class APIError(GrokError):
    """General API error for non-specific failures.

    Attributes:
        status_code: HTTP status code from the API
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details, error_code="api_error")
        self.status_code = status_code


class AuthenticationError(GrokError):
    """Exception raised for authentication failures.

    This indicates issues with API key or authorization.
    """

    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details, error_code="authentication_error")


class NetworkError(GrokError):
    """Exception raised for network-related failures.

    This includes timeouts, connection errors, and DNS failures.

    Attributes:
        original_error: The underlying network exception if available
    """

    def __init__(
        self,
        message: str = "Network error occurred",
        original_error: Optional[Exception] = None,
        details: Optional[dict[str, Any]] = None
    ) -> None:
        super().__init__(message, details, error_code="network_error")
        self.original_error = original_error
