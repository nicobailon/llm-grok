"""Tests for the HTTP client module."""

import json
from collections.abc import Iterator

import httpx
import pytest
from pytest_httpx import HTTPXMock

from llm_grok.client import GrokClient
from llm_grok.exceptions import (
    AuthenticationError,
)
from llm_grok.types import AnthropicMessage, Message
from tests.utils.mocks import (
    CHAT_COMPLETIONS_URL,
    ERROR_RESPONSES,
    MESSAGES_URL,
    TEST_API_KEY,
    create_chat_completion_response,
    create_messages_response,
)


class TestGrokClient:
    """Test cases for GrokClient class."""

    def test_client_initialization(self) -> None:
        """Test client initialization."""
        client = GrokClient(api_key=TEST_API_KEY)
        assert client.api_key == TEST_API_KEY
        assert client.timeout == 60.0  # Default timeout

    def test_successful_request(self, httpx_mock: HTTPXMock) -> None:
        """Test successful API request."""
        expected_response = create_chat_completion_response("Test response")

        httpx_mock.add_response(
            method="POST",
            url=CHAT_COMPLETIONS_URL,
            json=expected_response
        )

        client = GrokClient(api_key=TEST_API_KEY)
        messages: list[Message] = [{"role": "user", "content": "Hello"}]
        model = "x-ai/grok-4"
        stream = False

        response = client.post_openai_completion(
            messages=messages,
            model=model,
            stream=stream
        )

        # Response is not a stream, so we can access json() directly
        assert isinstance(response, httpx.Response)
        assert response.json() == expected_response

        # Verify request
        request = httpx_mock.get_request()
        assert request is not None
        assert request.headers["Authorization"] == f"Bearer {TEST_API_KEY}"
        request_body = json.loads(request.content)
        assert request_body["model"] == model
        assert request_body["messages"] == messages
        assert request_body["stream"] == stream
        assert request_body["temperature"] == 0.7  # Default temperature

    def test_rate_limit_retry(self, httpx_mock: HTTPXMock) -> None:
        """Test rate limit retry mechanism."""
        # First request fails with rate limit
        httpx_mock.add_response(
            method="POST",
            url=CHAT_COMPLETIONS_URL,
            status_code=429,
            json=ERROR_RESPONSES["rate_limit"],
            headers={"Retry-After": "1"}
        )

        # Second request succeeds
        expected_response = create_chat_completion_response("Success after retry")
        httpx_mock.add_response(
            method="POST",
            url=CHAT_COMPLETIONS_URL,
            json=expected_response
        )

        client = GrokClient(api_key=TEST_API_KEY)
        messages: list[Message] = []
        model = "x-ai/grok-4"

        # Should retry and succeed
        response = client.post_openai_completion(
            messages=messages,
            model=model
        )
        assert isinstance(response, httpx.Response)
        assert response.json() == expected_response

        # Verify two requests were made
        requests = httpx_mock.get_requests()
        assert len(requests) == 2

    def test_authentication_error(self, httpx_mock: HTTPXMock) -> None:
        """Test authentication error handling."""
        httpx_mock.add_response(
            method="POST",
            url=CHAT_COMPLETIONS_URL,
            status_code=401,
            json=ERROR_RESPONSES["authentication_error"]
        )

        client = GrokClient(api_key="invalid_key")
        messages: list[Message] = []
        model = "x-ai/grok-4"

        with pytest.raises(AuthenticationError) as exc_info:
            client.post_openai_completion(
                messages=messages,
                model=model
            )

        assert "Invalid API key" in str(exc_info.value)

    def test_streaming_request(self, httpx_mock: HTTPXMock) -> None:
        """Test streaming response handling."""
        # Mock streaming response chunks
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
            b'data: [DONE]\n\n'
        ]

        # Create a mock stream response
        def stream_iterator() -> Iterator[bytes]:
            yield from chunks

        httpx_mock.add_response(
            method="POST",
            url=CHAT_COMPLETIONS_URL,
            status_code=200,
            content=b"".join(chunks),
            headers={"content-type": "text/event-stream"}
        )

        client = GrokClient(api_key=TEST_API_KEY)

        # Make streaming request
        stream_response = client.post_openai_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="x-ai/grok-4",
            stream=True
        )
        # For streaming response, we need to handle context manager
        assert not isinstance(stream_response, httpx.Response)
        with stream_response:
            # In a real scenario, we'd parse SSE chunks here
            # For testing, just verify the request was made correctly
            pass

        # Verify the request
        request = httpx_mock.get_request()
        assert request is not None
        assert request.headers["Authorization"] == f"Bearer {TEST_API_KEY}"
        request_body = json.loads(request.content)
        assert request_body["model"] == "x-ai/grok-4"
        assert request_body["stream"] is True

    def test_post_anthropic_messages(self, httpx_mock: HTTPXMock) -> None:
        """Test Anthropic messages endpoint."""
        expected_response = create_messages_response("Test response")

        httpx_mock.add_response(
            method="POST",
            url=MESSAGES_URL,
            json=expected_response
        )

        client = GrokClient(api_key=TEST_API_KEY)
        messages: list[AnthropicMessage] = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        model = "x-ai/grok-4"
        max_tokens = 100

        response = client.post_anthropic_messages(
            request_data={"messages": messages},
            model=model,
            max_tokens=max_tokens
        )

        # Response is not a stream
        assert isinstance(response, httpx.Response)
        assert response.json() == expected_response

        # Verify request format
        request = httpx_mock.get_request()
        assert request is not None
        assert str(request.url) == MESSAGES_URL
        request_body = json.loads(request.content)
        assert request_body["model"] == model
        assert request_body["messages"] == messages
        assert request_body["max_tokens"] == max_tokens
        assert request_body["temperature"] == 0.7  # Default temperature
        assert request_body["stream"] is False  # Default stream value

    def test_error_parsing(self, httpx_mock: HTTPXMock) -> None:
        """Test error response parsing."""
        client = GrokClient(api_key=TEST_API_KEY)

        # Standard error format - create a mock response
        error_dict = {
            "error": {
                "message": "Test error message",
                "type": "invalid_request_error",
                "code": "test_error"
            }
        }

        # Create a mock response with error JSON
        httpx_mock.add_response(
            method="POST",
            url=CHAT_COMPLETIONS_URL,
            status_code=400,
            json=error_dict
        )

        # Make a request to trigger error handling
        with httpx.Client() as http_client:
            response = http_client.post(
                CHAT_COMPLETIONS_URL,
                json={"test": "data"},
                headers={"Authorization": f"Bearer {TEST_API_KEY}"}
            )

            # Parse the error response
            message, code = client._parse_error_response(response)
            assert message == "Test error message"
            assert code == "test_error"

        # Test with simple message format
        httpx_mock.add_response(
            method="POST",
            url=CHAT_COMPLETIONS_URL,
            status_code=400,
            json={"message": "Simple error message", "code": "simple_error"}
        )

        with httpx.Client() as http_client:
            response = http_client.post(
                CHAT_COMPLETIONS_URL,
                json={"test": "data"},
                headers={"Authorization": f"Bearer {TEST_API_KEY}"}
            )

            message, code = client._parse_error_response(response)
            assert message == "Simple error message"
            assert code == "simple_error"

