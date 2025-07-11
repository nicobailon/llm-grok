"""Tests for exception classes in llm_grok.exceptions module."""

import pytest
from llm_grok.exceptions import (
    GrokError,
    RateLimitError,
    QuotaExceededError,
    ValidationError,
    ConversionError,
    APIError,
    AuthenticationError,
    NetworkError,
)


class TestGrokError:
    """Test suite for base GrokError exception."""
    
    def test_basic_initialization(self) -> None:
        """Test basic GrokError initialization."""
        error = GrokError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details == {}
        assert error.error_code is None
    
    def test_with_details(self) -> None:
        """Test GrokError with details."""
        details = {"field": "temperature", "value": 2.0}
        error = GrokError("Invalid parameter", details=details)
        assert error.message == "Invalid parameter"
        assert error.details == details
        assert error.details["field"] == "temperature"
    
    def test_with_error_code(self) -> None:
        """Test GrokError with error code."""
        error = GrokError("API failure", error_code="api_error")
        assert error.error_code == "api_error"
        assert str(error) == "[api_error] API failure"
    
    def test_inheritance(self) -> None:
        """Test GrokError inherits from Exception."""
        error = GrokError("Test")
        assert isinstance(error, Exception)
        
        # Can be raised and caught
        with pytest.raises(GrokError) as exc_info:
            raise error
        assert exc_info.value.message == "Test"


class TestRateLimitError:
    """Test suite for RateLimitError exception."""
    
    def test_default_initialization(self) -> None:
        """Test RateLimitError with defaults."""
        error = RateLimitError()
        assert error.message == "Rate limit exceeded"
        assert error.error_code == "rate_limit_error"
        assert error.retry_after is None
        assert str(error) == "[rate_limit_error] Rate limit exceeded"
    
    def test_with_retry_after(self) -> None:
        """Test RateLimitError with retry_after."""
        error = RateLimitError(retry_after=60)
        assert error.retry_after == 60
    
    def test_custom_message(self) -> None:
        """Test RateLimitError with custom message."""
        error = RateLimitError("Too many requests", retry_after=30)
        assert error.message == "Too many requests"
        assert error.retry_after == 30
    
    def test_inheritance(self) -> None:
        """Test RateLimitError inherits from GrokError."""
        error = RateLimitError()
        assert isinstance(error, GrokError)
        assert isinstance(error, Exception)


class TestQuotaExceededError:
    """Test suite for QuotaExceededError exception."""
    
    def test_default_initialization(self) -> None:
        """Test QuotaExceededError with defaults."""
        error = QuotaExceededError()
        assert error.message == "API quota exceeded"
        assert error.error_code == "quota_exceeded_error"
        assert str(error) == "[quota_exceeded_error] API quota exceeded"
    
    def test_custom_message(self) -> None:
        """Test QuotaExceededError with custom message."""
        details = {"quota_limit": 1000, "current_usage": 1050}
        error = QuotaExceededError("Monthly quota exceeded", details=details)
        assert error.message == "Monthly quota exceeded"
        assert error.details["quota_limit"] == 1000
    
    def test_inheritance(self) -> None:
        """Test QuotaExceededError inherits from GrokError."""
        error = QuotaExceededError()
        assert isinstance(error, GrokError)


class TestValidationError:
    """Test suite for ValidationError exception."""
    
    def test_basic_validation_error(self) -> None:
        """Test basic ValidationError."""
        error = ValidationError("Invalid temperature value")
        assert error.message == "Invalid temperature value"
        assert error.error_code == "validation_error"
        assert error.field is None
    
    def test_with_field(self) -> None:
        """Test ValidationError with field information."""
        error = ValidationError("Value out of range", field="temperature")
        assert error.field == "temperature"
        assert str(error) == "[validation_error] Value out of range"
    
    def test_with_details(self) -> None:
        """Test ValidationError with additional details."""
        details = {"min": 0, "max": 1, "provided": 2}
        error = ValidationError(
            "Temperature out of range",
            field="temperature",
            details=details
        )
        assert error.field == "temperature"
        assert error.details["provided"] == 2
    
    def test_inheritance(self) -> None:
        """Test ValidationError inherits from GrokError."""
        error = ValidationError("Test")
        assert isinstance(error, GrokError)


class TestConversionError:
    """Test suite for ConversionError exception."""
    
    def test_basic_conversion_error(self) -> None:
        """Test basic ConversionError."""
        error = ConversionError("Failed to convert message format")
        assert error.message == "Failed to convert message format"
        assert error.error_code == "conversion_error"
        assert error.source_format is None
        assert error.target_format is None
    
    def test_with_formats(self) -> None:
        """Test ConversionError with format information."""
        error = ConversionError(
            "Cannot convert tool format",
            source_format="openai",
            target_format="anthropic"
        )
        assert error.source_format == "openai"
        assert error.target_format == "anthropic"
    
    def test_with_full_details(self) -> None:
        """Test ConversionError with all parameters."""
        details = {"field": "tool_calls", "reason": "unsupported structure"}
        error = ConversionError(
            "Tool conversion failed",
            source_format="openai",
            target_format="anthropic",
            details=details
        )
        assert error.source_format == "openai"
        assert error.target_format == "anthropic"
        assert error.details["reason"] == "unsupported structure"


class TestAPIError:
    """Test suite for APIError exception."""
    
    def test_basic_api_error(self) -> None:
        """Test basic APIError."""
        error = APIError("Server error occurred")
        assert error.message == "Server error occurred"
        assert error.error_code == "api_error"
        assert error.status_code is None
    
    def test_with_status_code(self) -> None:
        """Test APIError with HTTP status code."""
        error = APIError("Internal server error", status_code=500)
        assert error.status_code == 500
        assert str(error) == "[api_error] Internal server error"
    
    def test_with_details(self) -> None:
        """Test APIError with response details."""
        details = {"request_id": "req_123", "timestamp": "2024-01-01T00:00:00Z"}
        error = APIError("Bad request", status_code=400, details=details)
        assert error.status_code == 400
        assert error.details["request_id"] == "req_123"


class TestAuthenticationError:
    """Test suite for AuthenticationError exception."""
    
    def test_default_auth_error(self) -> None:
        """Test default AuthenticationError."""
        error = AuthenticationError()
        assert error.message == "Authentication failed"
        assert error.error_code == "authentication_error"
    
    def test_custom_auth_message(self) -> None:
        """Test AuthenticationError with custom message."""
        error = AuthenticationError("Invalid API key provided")
        assert error.message == "Invalid API key provided"
    
    def test_with_details(self) -> None:
        """Test AuthenticationError with details."""
        details = {"key_prefix": "xai-", "key_length": 32}
        error = AuthenticationError("API key format invalid", details=details)
        assert error.details["key_prefix"] == "xai-"


class TestNetworkError:
    """Test suite for NetworkError exception."""
    
    def test_default_network_error(self) -> None:
        """Test default NetworkError."""
        error = NetworkError()
        assert error.message == "Network error occurred"
        assert error.error_code == "network_error"
        assert error.original_error is None
    
    def test_with_original_error(self) -> None:
        """Test NetworkError wrapping another exception."""
        original = ConnectionError("Connection refused")
        error = NetworkError("Failed to connect to API", original_error=original)
        assert error.original_error is original
        assert isinstance(error.original_error, ConnectionError)
    
    def test_with_timeout_details(self) -> None:
        """Test NetworkError for timeout scenario."""
        import socket
        original = socket.timeout("Operation timed out")
        details = {"timeout": 30, "endpoint": "https://api.x.ai"}
        error = NetworkError(
            "Request timed out",
            original_error=original,
            details=details
        )
        assert error.original_error is original
        assert error.details["timeout"] == 30


class TestExceptionHierarchy:
    """Test the exception hierarchy relationships."""
    
    def test_all_inherit_from_grok_error(self) -> None:
        """Test all custom exceptions inherit from GrokError."""
        exceptions = [
            RateLimitError(),
            QuotaExceededError(),
            ValidationError("test"),
            ConversionError("test"),
            APIError("test"),
            AuthenticationError(),
            NetworkError(),
        ]
        
        for exc in exceptions:
            assert isinstance(exc, GrokError)
            assert isinstance(exc, Exception)
    
    def test_exception_catching(self) -> None:
        """Test exception catching patterns."""
        # Can catch specific exception
        with pytest.raises(RateLimitError):
            raise RateLimitError()
        
        # Can catch as GrokError
        with pytest.raises(GrokError):
            raise ValidationError("Invalid input")
        
        # Can catch as Exception
        with pytest.raises(Exception):
            raise NetworkError()
    
    def test_error_codes_unique(self) -> None:
        """Test each exception type has a unique error code."""
        error_codes = {
            GrokError("test").error_code,
            RateLimitError().error_code,
            QuotaExceededError().error_code,
            ValidationError("test").error_code,
            ConversionError("test").error_code,
            APIError("test").error_code,
            AuthenticationError().error_code,
            NetworkError().error_code,
        }
        
        # Remove None (from base GrokError)
        error_codes.discard(None)
        
        # All error codes should be unique
        assert len(error_codes) == 7  # One for each subclass