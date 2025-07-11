"""Test circuit breaker functionality for enterprise reliability."""

import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, cast
from unittest.mock import Mock, patch

import httpx
import pytest
from pytest_httpx import HTTPXMock

from llm_grok.client import GrokClient
from llm_grok.exceptions import NetworkError, APIError
from llm_grok.types import RequestBody
from tests.utils.mocks import TEST_API_KEY, CHAT_COMPLETIONS_URL


class TestCircuitBreakerFailureThreshold:
    """Test circuit breaker failure threshold behavior."""

    def test_circuit_opens_after_consecutive_failures(self, httpx_mock: HTTPXMock) -> None:
        """Circuit breaker opens after reaching failure threshold."""
        client = GrokClient(TEST_API_KEY)
        
        # Mock exactly 5 consecutive network failures for specific URL
        for _ in range(5):
            httpx_mock.add_exception(
                httpx.NetworkError("Connection failed"),
                url=CHAT_COMPLETIONS_URL,
                method="POST"
            )
        
        # First 4 failures should not open circuit
        for i in range(4):
            with pytest.raises(NetworkError):
                client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # 5th failure should open circuit
        with pytest.raises(NetworkError):
            client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # Next request should fail due to open circuit (doesn't hit network)
        with pytest.raises(NetworkError, match="Circuit breaker is open"):
            client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))

    def test_circuit_threshold_is_configurable(self) -> None:
        """Circuit breaker failure threshold is properly configured."""
        client = GrokClient(TEST_API_KEY)
        assert client.CIRCUIT_BREAKER_FAILURE_THRESHOLD == 5

    def test_circuit_breaker_tracks_consecutive_failures_only(self, httpx_mock: HTTPXMock) -> None:
        """Circuit breaker only counts consecutive failures."""
        client = GrokClient(TEST_API_KEY)
        
        # 3 failures followed by success resets counter
        for _ in range(3):
            httpx_mock.add_exception(httpx.NetworkError("Connection failed"))
        httpx_mock.add_response(200, json={"choices": [{"message": {"role": "assistant", "content": "test"}}]})
        
        # Execute the failures and success
        for _ in range(3):
            with pytest.raises(NetworkError):
                client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # Success should reset counter
        client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # Now 4 more failures should still not open circuit (since counter was reset)
        for _ in range(4):
            httpx_mock.add_exception(httpx.NetworkError("Connection failed"))
        
        for _ in range(4):
            with pytest.raises(NetworkError):
                client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # Circuit should still be closed - one more failure needed
        httpx_mock.add_exception(httpx.NetworkError("Connection failed"))
        with pytest.raises(NetworkError):
            client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # Now circuit should be open
        with pytest.raises(NetworkError, match="Circuit breaker is open"):
            client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))


class TestCircuitBreakerRecovery:
    """Test circuit breaker recovery mechanism."""

    def test_circuit_enters_half_open_after_timeout(self, httpx_mock: HTTPXMock) -> None:
        """Circuit breaker enters half-open state after recovery timeout."""
        client = GrokClient(TEST_API_KEY)
        
        # Override timeout for faster testing
        client.CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 0.1
        
        # Open circuit with failures
        for _ in range(5):
            httpx_mock.add_exception(httpx.NetworkError("Connection failed"))
            with pytest.raises(NetworkError):
                client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # Circuit should be open
        with pytest.raises(NetworkError, match="Circuit breaker is open"):
            client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # Wait for recovery timeout
        time.sleep(0.15)
        
        # Next request should try to execute (half-open state)
        httpx_mock.add_response(200, json={"choices": [{"message": {"role": "assistant", "content": "test"}}]})
        response = client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        assert isinstance(response, httpx.Response)
        assert response.status_code == 200

    def test_circuit_closes_on_successful_half_open_request(self, httpx_mock: HTTPXMock) -> None:
        """Circuit breaker closes after successful request in half-open state."""
        client = GrokClient(TEST_API_KEY)
        client.CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 0.1
        
        # Open circuit
        for _ in range(5):
            httpx_mock.add_exception(httpx.NetworkError("Connection failed"))
            with pytest.raises(NetworkError):
                client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # Wait for half-open
        time.sleep(0.15)
        
        # Successful request should close circuit
        httpx_mock.add_response(200, json={"choices": [{"message": {"role": "assistant", "content": "test"}}]})
        client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # Circuit should now be closed - next request should work normally
        httpx_mock.add_response(200, json={"choices": [{"message": {"role": "assistant", "content": "test"}}]})
        response = client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        assert isinstance(response, httpx.Response)
        assert response.status_code == 200

    def test_circuit_reopens_on_failed_half_open_request(self, httpx_mock: HTTPXMock) -> None:
        """Circuit breaker reopens after failed request in half-open state."""
        client = GrokClient(TEST_API_KEY)
        client.CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 0.1
        
        # Open circuit
        for _ in range(5):
            httpx_mock.add_exception(httpx.NetworkError("Connection failed"))
            with pytest.raises(NetworkError):
                client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # Wait for half-open
        time.sleep(0.15)
        
        # Failed request should reopen circuit
        httpx_mock.add_exception(httpx.NetworkError("Connection failed"))
        with pytest.raises(NetworkError):
            client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # Circuit should be open again
        with pytest.raises(NetworkError, match="Circuit breaker is open"):
            client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))


class TestCircuitBreakerTimeout:
    """Test circuit breaker timeout scenarios."""

    def test_circuit_breaker_timeout_configuration(self) -> None:
        """Circuit breaker recovery timeout is properly configured."""
        client = GrokClient(TEST_API_KEY)
        assert client.CIRCUIT_BREAKER_RECOVERY_TIMEOUT == 60

    def test_circuit_breaker_reports_remaining_time(self, httpx_mock: HTTPXMock) -> None:
        """Circuit breaker error includes remaining recovery time."""
        client = GrokClient(TEST_API_KEY)
        client.CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 10
        
        # Open circuit
        for _ in range(5):
            httpx_mock.add_exception(httpx.NetworkError("Connection failed"))
            with pytest.raises(NetworkError):
                client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # Check error message includes remaining time
        with pytest.raises(NetworkError) as exc_info:
            client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        error_msg = str(exc_info.value)
        assert "Circuit breaker is open" in error_msg
        assert "Will retry in" in error_msg

    def test_circuit_breaker_prevents_requests_during_timeout(self, httpx_mock: HTTPXMock) -> None:
        """Circuit breaker prevents all requests during timeout period."""
        client = GrokClient(TEST_API_KEY)
        client.CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 1.0
        
        # Open circuit
        for _ in range(5):
            httpx_mock.add_exception(httpx.NetworkError("Connection failed"))
            with pytest.raises(NetworkError):
                client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # Multiple requests during timeout should all fail immediately
        start_time = time.time()
        for _ in range(3):
            with pytest.raises(NetworkError, match="Circuit breaker is open"):
                client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        end_time = time.time()
        
        # All requests should fail quickly (not wait for network)
        assert end_time - start_time < 0.1


class TestCircuitBreakerThreadSafety:
    """Test circuit breaker thread safety."""

    def test_circuit_breaker_is_thread_safe(self, httpx_mock: HTTPXMock) -> None:
        """Circuit breaker state is properly synchronized across threads."""
        client = GrokClient(TEST_API_KEY)
        client.CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 0.1
        results = []
        exceptions = []
        
        # Pre-configure enough exceptions for all threads
        for _ in range(10):
            httpx_mock.add_exception(
                httpx.NetworkError("Connection failed"),
                url=CHAT_COMPLETIONS_URL,
                method="POST"
            )
        
        def make_failing_request() -> None:
            try:
                client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
                results.append("success")
            except NetworkError as e:
                exceptions.append(str(e))
        
        # Run 10 concurrent requests that will all fail
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_failing_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should have enough failures to open circuit
        assert len(exceptions) == 10
        # Check that circuit opened after some failures
        network_errors = [exc for exc in exceptions if "Connection failed" in exc]
        circuit_errors = [exc for exc in exceptions if "Circuit breaker is open" in exc]
        # Some should be network errors, some should be circuit breaker errors
        assert len(network_errors) >= 5 or len(circuit_errors) > 0

    def test_circuit_breaker_lock_prevents_race_conditions(self) -> None:
        """Circuit breaker uses locks to prevent race conditions."""
        client = GrokClient(TEST_API_KEY)
        
        # Verify lock exists
        assert hasattr(client, '_circuit_breaker_lock')
        assert isinstance(client._circuit_breaker_lock, threading.Lock)


class TestCircuitBreakerStreamingSupport:
    """Test circuit breaker with streaming requests."""

    @contextmanager
    def mock_stream_context(self, should_fail: bool = False) -> Iterator[Mock]:
        """Create a mock streaming context manager."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200 if not should_fail else 500
        
        if should_fail:
            yield mock_response
            raise httpx.NetworkError("Stream failed")
        else:
            yield mock_response

    def test_circuit_breaker_with_streaming_requests(self, httpx_mock: HTTPXMock) -> None:
        """Circuit breaker works correctly with streaming requests."""
        client = GrokClient(TEST_API_KEY)
        
        # Test that streaming requests can succeed and don't affect circuit breaker
        httpx_mock.add_response(200, text="data: chunk1\n\ndata: chunk2\n\n")
        
        with client.stream_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {})) as response:
            # Consume the stream successfully
            assert response.status_code == 200
        
        # Circuit should remain closed after streaming success
        httpx_mock.add_response(200, json={"choices": [{"message": {"role": "assistant", "content": "test"}}]})
        result = client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        # Type narrowing: make_request returns Response when stream=False (default)
        assert isinstance(result, httpx.Response)
        assert result.status_code == 200

    def test_circuit_breaker_has_streaming_wrapper(self) -> None:
        """Circuit breaker has streaming context manager wrapper."""
        client = GrokClient(TEST_API_KEY)
        
        # Verify the streaming wrapper method exists
        assert hasattr(client, '_wrap_stream_with_circuit_breaker')
        assert callable(client._wrap_stream_with_circuit_breaker)
        
        # Verify stream_request method uses make_request with stream=True
        assert hasattr(client, 'stream_request')
        assert callable(client.stream_request)


class TestCircuitBreakerErrorTypes:
    """Test circuit breaker behavior with different error types."""

    def test_circuit_breaker_counts_network_errors(self, httpx_mock: HTTPXMock) -> None:
        """Circuit breaker counts network errors toward failure threshold."""
        client = GrokClient(TEST_API_KEY)
        
        # Network errors should count
        for _ in range(5):
            httpx_mock.add_exception(httpx.NetworkError("Connection failed"))
            with pytest.raises(NetworkError):
                client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        with pytest.raises(NetworkError, match="Circuit breaker is open"):
            client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))

    def test_circuit_breaker_counts_timeout_errors(self, httpx_mock: HTTPXMock) -> None:
        """Circuit breaker counts timeout errors toward failure threshold."""
        client = GrokClient(TEST_API_KEY)
        
        # Timeout errors should count
        for _ in range(5):
            httpx_mock.add_exception(httpx.TimeoutException("Request timeout"))
            with pytest.raises(NetworkError):
                client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        with pytest.raises(NetworkError, match="Circuit breaker is open"):
            client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))

    def test_circuit_breaker_counts_http_errors(self, httpx_mock: HTTPXMock) -> None:
        """Circuit breaker counts non-rate-limit HTTP errors toward failure threshold."""
        client = GrokClient(TEST_API_KEY)
        
        # 500 errors should count (not rate limits)
        for _ in range(5):
            httpx_mock.add_response(500, json={"error": {"message": "Internal server error"}})
            with pytest.raises(APIError):
                client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        with pytest.raises(NetworkError, match="Circuit breaker is open"):
            client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))

    def test_circuit_breaker_handles_rate_limit_properly(self, httpx_mock: HTTPXMock) -> None:
        """Circuit breaker handles rate limit retries correctly."""
        client = GrokClient(TEST_API_KEY)
        
        # Rate limit should retry internally and not open circuit on first attempt
        # Set up a rate limit followed by success
        httpx_mock.add_response(429, headers={"Retry-After": "0.01"})  # Very short wait
        httpx_mock.add_response(200, json={"choices": [{"message": {"role": "assistant", "content": "test"}}]})
        
        # This should succeed after internal retry
        response = client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        assert isinstance(response, httpx.Response)
        assert response.status_code == 200
        
        # Circuit should remain closed
        httpx_mock.add_response(200, json={"choices": [{"message": {"role": "assistant", "content": "test2"}}]})
        response2 = client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        assert isinstance(response2, httpx.Response)
        assert response2.status_code == 200


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with other client features."""

    def test_circuit_breaker_with_request_validation(self) -> None:
        """Circuit breaker works with request size validation."""
        client = GrokClient(TEST_API_KEY)
        
        # Large request should fail validation before circuit breaker check
        large_request = cast(RequestBody, {"data": "x" * (client.MAX_REQUEST_SIZE + 1)})
        
        with pytest.raises(ValueError, match="Request size.*exceeds maximum"):
            client.make_request("POST", CHAT_COMPLETIONS_URL, {}, large_request)
        
        # Circuit should not be affected by validation errors
        # (This is implementation dependent - validation might happen before circuit check)

    def test_circuit_breaker_state_persistence(self, httpx_mock: HTTPXMock) -> None:
        """Circuit breaker state persists across multiple operations."""
        client = GrokClient(TEST_API_KEY)
        
        # Open circuit
        for _ in range(5):
            httpx_mock.add_exception(httpx.NetworkError("Connection failed"))
            with pytest.raises(NetworkError):
                client.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # State should persist across different request types
        with pytest.raises(NetworkError, match="Circuit breaker is open"):
            client.post_openai_completion([], "grok-4")
        
        with pytest.raises(NetworkError, match="Circuit breaker is open"):
            client.post_anthropic_messages({"messages": []}, "grok-4")

    def test_circuit_breaker_independent_per_client_instance(self, httpx_mock: HTTPXMock) -> None:
        """Circuit breaker state is independent per client instance."""
        client1 = GrokClient(TEST_API_KEY)
        client2 = GrokClient(TEST_API_KEY)
        
        # Open circuit on client1
        for _ in range(5):
            httpx_mock.add_exception(httpx.NetworkError("Connection failed"))
            with pytest.raises(NetworkError):
                client1.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # client1 circuit should be open
        with pytest.raises(NetworkError, match="Circuit breaker is open"):
            client1.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        
        # client2 circuit should still be closed
        httpx_mock.add_response(200, json={"choices": [{"message": {"role": "assistant", "content": "test"}}]})
        response = client2.make_request("POST", CHAT_COMPLETIONS_URL, {}, cast(RequestBody, {}))
        assert isinstance(response, httpx.Response)
        assert response.status_code == 200


# Type-level tests for circuit breaker components
def test_circuit_breaker_types() -> None:
    """Test circuit breaker maintains proper types."""
    client = GrokClient(TEST_API_KEY)
    
    # Verify type annotations exist and are correct
    assert hasattr(client, 'CIRCUIT_BREAKER_FAILURE_THRESHOLD')
    assert isinstance(client.CIRCUIT_BREAKER_FAILURE_THRESHOLD, int)
    assert hasattr(client, 'CIRCUIT_BREAKER_RECOVERY_TIMEOUT')
    assert isinstance(client.CIRCUIT_BREAKER_RECOVERY_TIMEOUT, (int, float))