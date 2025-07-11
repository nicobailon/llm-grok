"""HTTP client for Grok API with retry logic and rate limit handling."""

import hashlib
import json
import logging
import random
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Literal, Optional, Union, cast
from typing_extensions import TypeGuard
from contextlib import AbstractContextManager

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .constants import (
    DEFAULT_KEEPALIVE_RATIO,
    DEFAULT_MAX_CONNECTIONS,
    DEFAULT_TIMEOUT,
    JITTER_FACTOR_MAX,
    JITTER_FACTOR_MIN,
    SLEEP_INTERVAL_MAX,
)
from .exceptions import (
    APIError,
    AuthenticationError,
    GrokError,
    NetworkError,
    QuotaExceededError,
    RateLimitError,
)
from .types import (
    AnthropicRequest,
    AnthropicResponse,
    AnthropicToolChoice,
    AnthropicToolDefinition,
    Message,
    OpenAIResponse,
    RequestBody,
    ResponseFormat,
    StreamOptions,
    ToolChoice,
    ToolDefinition,
)

console = Console()
logger = logging.getLogger(__name__)


class GrokClient:
    """HTTP client for interacting with Grok API endpoints.
    
    Handles:
    - HTTP request execution with retry logic
    - Rate limit handling with exponential backoff
    - Error parsing and exception handling
    - Connection pooling and timeouts
    - Request/response logging hooks
    """

    # Constants
    MAX_RETRIES = 3
    BASE_DELAY = 1.0  # seconds
    API_URL = "https://api.x.ai/v1/chat/completions"
    MESSAGES_URL = "https://api.x.ai/v1/messages"

    # Resource limits for security
    MAX_WAIT_TIME = 60  # seconds - maximum time to wait for rate limiting
    MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB - maximum request body size
    MAX_BUFFER_SIZE = 100 * 1024 * 1024  # 100MB - maximum streaming buffer size

    # Circuit breaker settings
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5  # Number of consecutive failures before opening circuit
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: float = 60  # seconds to wait before attempting recovery
    CIRCUIT_BREAKER_HALF_OPEN_REQUESTS: int = 1  # Number of test requests in half-open state

    def __init__(
        self,
        api_key: str,
        timeout: Optional[float] = None,
        max_connections: Optional[int] = None,
        enable_logging: bool = False,
    ):
        """Initialize the Grok HTTP client.
        
        Args:
            api_key: xAI API key for authentication
            timeout: Request timeout in seconds (None for no timeout)
            max_connections: Maximum number of concurrent connections
            enable_logging: Enable request/response logging
        """
        self.api_key = api_key
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.max_connections = max_connections or DEFAULT_MAX_CONNECTIONS
        self.enable_logging = enable_logging

        # Create persistent client pool with connection pooling
        self._client_pool = httpx.Client(
            limits=httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=int(self.max_connections * DEFAULT_KEEPALIVE_RATIO),
            ),
            timeout=httpx.Timeout(self.timeout),
        )
        self._is_closed = False

        # Circuit breaker state with thread safety
        self._circuit_breaker_lock = threading.Lock()
        self._consecutive_failures = 0
        self._circuit_opened_at: Optional[float] = None
        self._half_open_requests = 0

        # JSON size estimation cache with memory monitoring
        self._json_size_cache: dict[str, int] = {}
        self._cache_lock = threading.Lock()
        self.MAX_CACHE_SIZE: int = 100  # Limit cache size to prevent memory issues
        self.MAX_CACHE_MEMORY: int = 1024 * 1024  # 1MB max cache memory usage
        self._cache_memory_usage = 0  # Track current cache memory usage

    def __enter__(self) -> "GrokClient":
        """Enter context manager for proper resource management."""
        self._client_pool.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager and cleanup resources."""
        if not self._is_closed:
            self._client_pool.__exit__(*args)
            self._cleanup_cache()
            self._is_closed = True

    def close(self) -> None:
        """Manually close the client pool."""
        if not self._is_closed:
            self._client_pool.close()
            self._cleanup_cache()
            self._is_closed = True

    def _cleanup_cache(self) -> None:
        """Clean up the JSON size cache and reset memory tracking."""
        with self._cache_lock:
            self._json_size_cache.clear()
            self._cache_memory_usage = 0

    def _log_request(self, method: str, url: str, headers: dict[str, str], body: RequestBody) -> None:
        """Log outgoing request details."""
        if not self.enable_logging:
            return

        console.print(f"[blue]→ {method} {url}[/blue]")
        # Don't log auth headers
        safe_headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
        console.print(f"[dim]Headers: {safe_headers}[/dim]")
        # Don't log full message content, just structure
        if "messages" in body and isinstance(body["messages"], list):
            console.print(f"[dim]Messages: {len(body['messages'])} messages[/dim]")
        console.print(f"[dim]Model: {body.get('model', 'unknown')}[/dim]")

    def _is_openai_response(self, response: dict[str, Any]) -> TypeGuard[OpenAIResponse]:
        """Type guard to check if response is OpenAI format."""
        return (
            "choices" in response 
            and isinstance(response.get("choices"), list)
            and "object" in response
        )

    def _is_anthropic_response(self, response: dict[str, Any]) -> TypeGuard[AnthropicResponse]:
        """Type guard to check if response is Anthropic format."""
        return (
            "content" in response 
            and isinstance(response.get("content"), list)
            and response.get("type") == "message"
        )

    def _log_response(self, status_code: int, response_data: Optional[Union[OpenAIResponse, AnthropicResponse, dict[str, str]]] = None) -> None:
        """Log incoming response details."""
        if not self.enable_logging:
            return

        console.print(f"[green]← {status_code}[/green]")
        if response_data and isinstance(response_data, dict):
            # Cast to dict[str, Any] for type guards
            response_dict = dict(response_data)
            if self._is_openai_response(response_dict):
                console.print(f"[dim]Choices: {len(response_dict['choices'])}[/dim]")
                if response_dict["choices"]:
                    choice = response_dict["choices"][0]
                    if "message" in choice:
                        console.print(f"[dim]Role: {choice['message'].get('role', 'unknown')}[/dim]")
            elif self._is_anthropic_response(response_dict):
                console.print(f"[dim]Content blocks: {len(response_dict['content'])}[/dim]")
                console.print(f"[dim]Role: {response_dict['role']}[/dim]")
            else:
                console.print("[dim]Unknown response format[/dim]")

    def _compute_cache_key(self, json_data: RequestBody) -> str:
        """Compute a cache key for JSON data.
        
        Args:
            json_data: JSON payload
            
        Returns:
            Cache key based on content hash
        """
        # Create a stable string representation of the data
        # We use sorted keys to ensure consistent hashing
        stable_str = json.dumps(json_data, sort_keys=True)
        return hashlib.md5(stable_str.encode('utf-8')).hexdigest()

    def _estimate_json_size(self, json_data: RequestBody) -> int:
        """Estimate JSON size with caching.
        
        Args:
            json_data: JSON payload
            
        Returns:
            Estimated size in bytes
        """
        cache_key = self._compute_cache_key(json_data)

        # Check cache first
        with self._cache_lock:
            if cache_key in self._json_size_cache:
                return self._json_size_cache[cache_key]

        # Compute actual size
        request_size = len(json.dumps(json_data).encode('utf-8'))

        # Update cache with size and memory limits
        with self._cache_lock:
            # Estimate memory usage of the cache key and value
            key_memory = len(cache_key.encode('utf-8'))
            value_memory = 8  # int size in Python (approximate)
            entry_memory = key_memory + value_memory
            
            # If adding this entry would exceed memory limit or size limit, clear cache
            if (self._cache_memory_usage + entry_memory > self.MAX_CACHE_MEMORY or 
                len(self._json_size_cache) >= self.MAX_CACHE_SIZE):
                self._json_size_cache.clear()
                self._cache_memory_usage = 0
            
            # Add entry and update memory tracking
            if cache_key not in self._json_size_cache:
                self._cache_memory_usage += entry_memory
            self._json_size_cache[cache_key] = request_size

        return request_size

    def _validate_request_size(self, json_data: RequestBody) -> None:
        """Validate that request data doesn't exceed size limits.
        
        Args:
            json_data: JSON payload to validate
            
        Raises:
            ValueError: If request size exceeds limits
        """
        # Use cached estimation
        request_size = self._estimate_json_size(json_data)

        if request_size > self.MAX_REQUEST_SIZE:
            raise ValueError(
                f"Request size ({request_size} bytes) exceeds maximum allowed size "
                f"({self.MAX_REQUEST_SIZE} bytes). Please reduce the size of your request."
            )

    def _calculate_backoff_delay(self, attempt: int, retry_after: Optional[int] = None) -> float:
        """Calculate backoff delay with exponential backoff and jitter.
        
        Args:
            attempt: Current retry attempt number (0-based)
            retry_after: Optional server-provided retry delay in seconds
            
        Returns:
            Delay in seconds with jitter applied
        """
        if retry_after is not None:
            # Use server-provided delay with small jitter
            base_delay = float(retry_after)
            jitter = random.uniform(0, JITTER_FACTOR_MIN * base_delay)
        else:
            # Exponential backoff: 1s, 2s, 4s, etc.
            base_delay = self.BASE_DELAY * (2 ** attempt)
            # Add jitter up to 25% of base delay
            jitter = random.uniform(0, JITTER_FACTOR_MAX * base_delay)

        delay = base_delay + jitter

        # Cap total delay
        if delay > self.MAX_WAIT_TIME:
            logger.warning(
                f"Calculated delay {delay:.2f}s exceeds maximum {self.MAX_WAIT_TIME}s, capping"
            )
            delay = self.MAX_WAIT_TIME

        return delay

    def _check_circuit_breaker(self) -> None:
        """Check circuit breaker state and raise error if circuit is open.
        
        Raises:
            NetworkError: If circuit is open and not ready for recovery
        """
        with self._circuit_breaker_lock:
            if self._circuit_opened_at is not None:
                elapsed = time.time() - self._circuit_opened_at

                if elapsed < self.CIRCUIT_BREAKER_RECOVERY_TIMEOUT:
                    # Circuit is still open
                    remaining = self.CIRCUIT_BREAKER_RECOVERY_TIMEOUT - elapsed
                    raise NetworkError(
                        f"Circuit breaker is open due to repeated failures. "
                        f"Will retry in {remaining:.0f} seconds."
                    )
                else:
                    # Move to half-open state
                    logger.info("Circuit breaker moving to half-open state")
                    self._half_open_requests = 0

    def _record_success(self) -> None:
        """Record a successful request for circuit breaker."""
        with self._circuit_breaker_lock:
            if self._circuit_opened_at is not None:
                # Success in half-open state - close circuit
                logger.info("Circuit breaker closing after successful request")
                self._circuit_opened_at = None
                self._consecutive_failures = 0
                self._half_open_requests = 0
            else:
                # Reset failure count on success
                self._consecutive_failures = 0

    def _record_failure(self) -> None:
        """Record a failed request for circuit breaker."""
        with self._circuit_breaker_lock:
            self._consecutive_failures += 1

            if self._circuit_opened_at is not None:
                # Failure in half-open state - reopen circuit
                logger.warning("Circuit breaker reopening after failure in half-open state")
                self._circuit_opened_at = time.time()
                self._half_open_requests = 0
            elif self._consecutive_failures >= self.CIRCUIT_BREAKER_FAILURE_THRESHOLD:
                # Open circuit after threshold exceeded
                logger.error(
                    f"Circuit breaker opening after {self._consecutive_failures} consecutive failures"
                )
                self._circuit_opened_at = time.time()

    @contextmanager
    def _wrap_stream_with_circuit_breaker(
        self, stream_cm: AbstractContextManager[httpx.Response]
    ) -> Iterator[httpx.Response]:
        """Wrap streaming context manager to track circuit breaker success/failure.
        
        Args:
            stream_cm: The original streaming context manager
            
        Yields:
            The streaming response
            
        Raises:
            Any exception from the stream, after recording failure
        """
        stream_succeeded = False
        try:
            with stream_cm as response:
                # Stream entered successfully, but we don't record success yet
                # We need to track if the stream is consumed without errors
                try:
                    yield response
                    # If we reach here, the stream was consumed successfully
                    stream_succeeded = True
                except Exception:
                    # Stream consumption failed
                    raise
        except Exception:
            # Either entering the stream or consuming it failed
            self._record_failure()
            raise
        finally:
            # Only record success if stream was consumed without exceptions
            if stream_succeeded:
                self._record_success()

    def _parse_error_response(self, response: httpx.Response) -> tuple[str, Optional[str]]:
        """Parse error details from API response.
        
        Args:
            response: HTTP response object
            
        Returns:
            Tuple of (error_message, error_code)
        """
        try:
            if response.is_stream_consumed:
                error_body = response.text
            else:
                error_body = response.read().decode("utf-8")
        except (AttributeError, UnicodeDecodeError, OSError):
            return str(response.status_code), None

        try:
            error_json = json.loads(error_body)

            # Check for standard error format
            if "error" in error_json:
                error_data = error_json["error"]
                message = error_data.get("message", "Unknown error")
                code = error_data.get("code")
                return message, code

            # Check for simple message format
            if "message" in error_json:
                return error_json["message"], error_json.get("code")

            # Return full body if no standard format
            return error_body, None

        except (json.JSONDecodeError, KeyError, TypeError):
            return error_body or str(response.status_code), None

    def _handle_rate_limit(self, response: httpx.Response, attempt: int) -> bool:
        """Handle rate limit response with retry logic.
        
        Args:
            response: HTTP response with 429 status
            attempt: Current retry attempt number
            
        Returns:
            True if should retry, False otherwise
            
        Raises:
            RateLimitError: If max retries exceeded
            QuotaExceededError: If quota is exhausted
        """
        retry_after = response.headers.get("Retry-After")

        # Check if this is a quota exceeded error
        error_message, error_code = self._parse_error_response(response)
        if "quota" in error_message.lower() or error_code == "quota_exceeded":
            raise QuotaExceededError(
                "API quota exceeded. Please check your usage at https://console.x.ai",
                details={"error": error_message}
            )

        if attempt >= self.MAX_RETRIES - 1:
            raise RateLimitError(
                f"Rate limit exceeded after {self.MAX_RETRIES} attempts",
                retry_after=int(retry_after) if retry_after else None
            )

        # Calculate wait time with jitter
        retry_after_int = None
        if retry_after:
            try:
                retry_after_int = int(retry_after)
            except ValueError:
                logger.warning(f"Invalid Retry-After header value: {retry_after}")

        wait_time = self._calculate_backoff_delay(attempt, retry_after_int)

        logger.info(
            f"Rate limited, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{self.MAX_RETRIES})"
        )

        # Show progress while waiting
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Rate limit hit. Waiting {wait_time}s...",
                total=wait_time,
            )
            start_time = time.time()
            # Use longer sleep intervals to reduce CPU usage
            sleep_interval = min(SLEEP_INTERVAL_MAX, wait_time)
            while time.time() - start_time < wait_time:
                remaining = wait_time - (time.time() - start_time)
                actual_sleep = min(sleep_interval, remaining)
                if actual_sleep > 0:
                    time.sleep(actual_sleep)
                    progress.update(task, advance=actual_sleep)

        return True

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Handle non-rate-limit error responses.
        
        Args:
            response: HTTP error response
            
        Raises:
            AuthenticationError: For 401 errors
            APIError: For other API errors
        """
        error_message, error_code = self._parse_error_response(response)

        if response.status_code == 401:
            raise AuthenticationError(
                "Invalid API key. Please check your xAI API key.",
                details={"error": error_message}
            )

        # Generic API error
        raise APIError(
            f"API request failed: {error_message}",
            status_code=response.status_code,
            details={"error_code": error_code} if error_code else None
        )


    def make_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        json_data: RequestBody,
        stream: bool = False,
    ) -> Union[httpx.Response, AbstractContextManager[httpx.Response]]:
        """Execute HTTP request with retry logic for rate limiting.
        
        Args:
            method: HTTP method (e.g., 'POST')
            url: Request URL
            headers: Request headers dict
            json_data: JSON payload for request body
            stream: Whether to stream the response
            
        Returns:
            httpx.Response or stream context manager
            
        Raises:
            RateLimitError: When rate limit is exceeded after all retries
            QuotaExceededError: When API quota is exceeded
            AuthenticationError: When API key is invalid
            APIError: For other API errors
            NetworkError: For network-related errors
            GrokError: For other errors
        """
        # Check circuit breaker state
        self._check_circuit_breaker()

        # Validate request size
        self._validate_request_size(json_data)

        self._log_request(method, url, headers, json_data)

        for attempt in range(self.MAX_RETRIES):
            try:
                if self._is_closed:
                    raise GrokError("Client has been closed")

                if stream:
                    # For streaming, wrap the context manager to track success/failure
                    stream_cm = self._client_pool.stream(
                        method, url, headers=headers, json=json_data
                    )
                    return self._wrap_stream_with_circuit_breaker(stream_cm)
                else:
                    response = self._client_pool.request(
                        method, url, headers=headers, json=json_data
                    )
                    response.raise_for_status()
                    self._log_response(response.status_code)
                    self._record_success()
                    return response

            except httpx.HTTPStatusError as e:
                # Handle HTTP errors with status codes
                if e.response.status_code == 429:
                    if self._handle_rate_limit(e.response, attempt):
                        continue  # Retry after rate limit wait
                else:
                    self._record_failure()
                    self._handle_error_response(e.response)

            except httpx.TimeoutException as e:
                self._record_failure()
                logger.error(f"Request timeout after {self.timeout}s")
                raise NetworkError(
                    "Request timed out",
                    original_error=e
                )

            except httpx.NetworkError as e:
                self._record_failure()
                logger.error(f"Network error: {str(e)}")
                raise NetworkError(
                    f"Network error: {str(e)}",
                    original_error=e
                )

            except Exception as e:
                self._record_failure()
                logger.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
                raise GrokError(
                    f"Unexpected error: {str(e)}",
                    details={"exception_type": type(e).__name__}
                )

        # Should not reach here due to exceptions in loop
        raise GrokError("Request failed after all retries")

    def stream_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        json_data: RequestBody,
    ) -> AbstractContextManager[httpx.Response]:
        """Execute streaming HTTP request with retry logic.
        
        This is a convenience method that wraps make_request with stream=True.
        
        Args:
            method: HTTP method (e.g., 'POST')
            url: Request URL
            headers: Request headers dict
            json_data: JSON payload for request body
            
        Returns:
            Stream context manager for reading response
            
        Raises:
            Same as make_request
        """
        result = self.make_request(method, url, headers, json_data, stream=True)
        # Type narrowing for mypy
        assert not isinstance(result, httpx.Response)
        return result

    def post_openai_completion(
        self,
        messages: list[Message],
        model: str,
        stream: bool = False,
        temperature: float = 0.7,
        max_completion_tokens: Optional[int] = None,
        tools: Optional[list[ToolDefinition]] = None,
        tool_choice: Optional[Union[Literal["auto", "none"], ToolChoice]] = None,
        response_format: Optional[ResponseFormat] = None,
        reasoning_effort: Optional[str] = None,
        stream_options: Optional[StreamOptions] = None,
    ) -> Union[httpx.Response, AbstractContextManager[httpx.Response]]:
        """Make a request to the OpenAI-compatible chat completions endpoint.
        
        Args:
            messages: List of conversation messages
            model: Model ID to use
            stream: Whether to stream the response
            temperature: Sampling temperature (0-1)
            max_completion_tokens: Maximum tokens to generate
            tools: Function/tool definitions
            tool_choice: Tool selection strategy
            response_format: Output format specification
            reasoning_effort: Reasoning depth control
            stream_options: Streaming configuration
            
        Returns:
            Response object or stream context manager
        """
        body = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
        }

        if max_completion_tokens is not None:
            body["max_completion_tokens"] = max_completion_tokens

        if tools is not None:
            body["tools"] = tools

        if tool_choice is not None:
            body["tool_choice"] = tool_choice

        if response_format is not None:
            body["response_format"] = response_format

        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort

        if stream and stream_options is not None:
            body["stream_options"] = stream_options

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        return self.make_request("POST", self.API_URL, headers, cast(RequestBody, body), stream=stream)

    def post_anthropic_messages(
        self,
        request_data: AnthropicRequest,
        model: str,
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[list[AnthropicToolDefinition]] = None,
        tool_choice: Optional[Union[Literal["auto", "none"], AnthropicToolChoice]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Union[httpx.Response, AbstractContextManager[httpx.Response]]:
        """Make a request to the Anthropic-compatible messages endpoint.
        
        Args:
            request_data: Anthropic format request with messages and optional system
            model: Model ID to use  
            stream: Whether to stream the response
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            tools: Function/tool definitions in Anthropic format
            tool_choice: Tool selection in Anthropic format
            reasoning_effort: Reasoning depth control
            
        Returns:
            Response object or stream context manager
        """
        body = {
            "model": model,
            "stream": stream,
            "temperature": temperature,
            **request_data  # Includes messages and optional system
        }

        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        if tools is not None:
            body["tools"] = tools

        if tool_choice is not None:
            body["tool_choice"] = tool_choice

        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        return self.make_request("POST", self.MESSAGES_URL, headers, cast(RequestBody, body), stream=stream)


__all__ = ["GrokClient"]
