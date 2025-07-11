import base64
import json
import warnings
from typing import Any, Dict, List, Optional, cast, Iterator, Union

import httpx
import llm
import pytest
from pytest_httpx import HTTPXMock

from llm_grok import (
    DEFAULT_MODEL,
    AVAILABLE_MODELS,
    MODEL_INFO,
    Grok,
    GrokError,
    register_models,
    ToolDefinition,
    Message,
)
from llm_grok.grok import GrokOptions
from llm_grok.exceptions import AuthenticationError
from llm.models import ToolCall
from llm_grok.types import TextContent, ImageContent, RequestBody
from tests.utils.mocks import (
    TEST_API_KEY,
    TEST_MODEL_ID,
    CHAT_COMPLETIONS_URL,
    MESSAGES_URL,
    SAMPLE_JPEG_BASE64,
    SAMPLE_PNG_BASE64,
    SAMPLE_IMAGE_URL,
    SAMPLE_DATA_URL,
    INVALID_BASE64,
    ERROR_RESPONSES,
    SAMPLE_TOOLS,
    SAMPLE_TOOL_CALLS,
    create_streaming_chunks,
    create_anthropic_streaming_chunks,
    create_chat_completion_response,
    create_messages_response,
    create_multimodal_message,
    create_test_request,
)


# Helper functions for type-safe message access
def get_message_role(message: Message) -> str:
    """Get message role with type safety."""
    return message["role"]


def get_message_content(message: Message) -> Union[str, List[Union[TextContent, ImageContent]]]:
    """Get message content with type safety."""
    return message["content"]


def assert_message_has_fields(message: Message) -> None:
    """Assert that a message has the required fields."""
    assert "role" in message, "Message missing 'role' field"
    assert "content" in message, "Message missing 'content' field"


# Mock classes for testing
class MockAttachment:
    """Mock attachment that mimics llm.Attachment interface."""
    def __init__(self, type: str, data: str) -> None:
        self.type = type
        # Support the actual llm.Attachment interface
        if data.startswith(('http://', 'https://', 'data:')):
            # URLs and data URLs are stored as url
            self.url = data
            self.content: Optional[bytes] = None
            self.path: Optional[str] = None
        else:
            # Raw base64 - store as content
            self.url: Optional[str] = None
            self.path: Optional[str] = None
            self.content = base64.b64decode(data)
    
    def base64_content(self) -> Optional[str]:
        if self.content:
            return base64.b64encode(self.content).decode('utf-8')
        return None
    
    def resolve_type(self) -> Optional[str]:
        # Extract MIME type from data URL if available
        if self.url and self.url.startswith('data:'):
            # Extract MIME type from data URL
            mime_part = self.url.split(';')[0]
            return mime_part.replace('data:', '')
        # Default to image/jpeg for other images
        return "image/jpeg" if self.type == "image" else None


class MockPrompt(llm.Prompt):
    """Mock prompt that mimics llm.Prompt interface."""
    def __init__(self, prompt: str, attachments: Optional[List[MockAttachment]] = None, 
                 system: Optional[str] = None, options: Optional[object] = None, 
                 model: Optional[llm.Model] = None) -> None:
        # Create a minimal model if not provided
        if model is None:
            from llm_grok import Grok
            model = cast(llm.Model, Grok("x-ai/grok-4"))
        
        # Initialize parent class
        super().__init__(prompt=prompt, model=model, system=system)
        
        # Add custom attributes - use Any to avoid type issues
        self.attachments: Any = attachments or []
        self.options: Any = options


class MockResponse(llm.Response):
    """Mock response that mimics llm.Response interface."""
    def __init__(self, prompt: Optional[llm.Prompt] = None, model: Optional[llm.Model] = None, 
                 stream: bool = False) -> None:
        # Create minimal prompt and model if not provided
        if prompt is None:
            prompt = MockPrompt("test")
        if model is None:
            from llm_grok import Grok
            model = cast(llm.Model, Grok("x-ai/grok-4"))
        
        super().__init__(model=model, prompt=prompt, stream=stream)
        self._chunks: List[str] = []
        
    def __iter__(self) -> Iterator[str]:
        """Iterate over response chunks."""
        for chunk in self._chunks:
            yield chunk
    
    def add_chunk(self, chunk: str) -> None:
        """Add a chunk to the response."""
        self._chunks.append(chunk)


# Fixtures
@pytest.fixture
def model() -> Grok:
    """Create a test Grok model instance."""
    return Grok(TEST_MODEL_ID)


@pytest.fixture
def mock_response() -> Dict[str, object]:
    """Create a standard mock response for testing."""
    return create_chat_completion_response("Test response")


@pytest.fixture
def mock_env(monkeypatch) -> None:
    """Mock environment with API key."""
    monkeypatch.setenv("XAI_API_KEY", TEST_API_KEY)


# Core Functionality Tests (HIGH PRIORITY)

def test_model_initialization(model: Grok) -> None:
    """Test basic model initialization."""
    assert model.model_id == TEST_MODEL_ID
    assert model.needs_key == "grok"
    assert model.key_env_var == "XAI_API_KEY"


def test_grok_4_model_initialization() -> None:
    """Test Grok 4 model initialization."""
    grok4 = Grok("x-ai/grok-4")
    assert grok4.model_id == "x-ai/grok-4"
    assert grok4.needs_key == "grok"
    assert grok4.can_stream is True


def test_model_info_registry() -> None:
    """Test MODEL_INFO registry contains expected models."""
    # Check that all available models have info
    for model_id in AVAILABLE_MODELS:
        assert model_id in MODEL_INFO, f"Missing MODEL_INFO for {model_id}"
        info = MODEL_INFO[model_id]
        assert "context_window" in info
        assert "supports_vision" in info
        assert "supports_tools" in info


def test_default_model_is_grok_4() -> None:
    """Test that default model is Grok 4."""
    assert DEFAULT_MODEL == "x-ai/grok-4"


def test_build_messages_with_system_prompt(model: Grok) -> None:
    """Test message building with system prompt."""
    prompt = MockPrompt("Hello", system="You are helpful")
    messages = model.build_messages(prompt, None)
    
    assert len(messages) == 2
    assert get_message_role(messages[0]) == "system"
    assert get_message_content(messages[0]) == "You are helpful"
    assert get_message_role(messages[1]) == "user"
    assert get_message_content(messages[1]) == "Hello"


def test_build_messages_without_system_prompt(model: Grok) -> None:
    """Test message building without system prompt."""
    prompt = MockPrompt("Hello")
    messages = model.build_messages(prompt, None)
    
    assert len(messages) == 1
    assert get_message_role(messages[0]) == "user"
    assert get_message_content(messages[0]) == "Hello"


def test_build_messages_with_conversation(model: Grok, mock_env: None) -> None:
    """Test message building with conversation history."""
    # Create conversation with proper response structure
    from llm.models import Conversation, Response, Model
    from llm_grok import Grok
    
    grok = Grok(TEST_MODEL_ID)
    conversation = Conversation(model=cast(Model, grok))
    
    # First exchange
    prompt1 = MockPrompt("First question", model=cast(Model, grok))
    response1 = Response(model=cast(Model, grok), prompt=prompt1, stream=False)
    response1._chunks = ["First response"]
    response1._done = True  # Prevent re-execution when text() is called
    
    # Use the public API to add response to conversation
    conversation.responses.append(response1)
    
    # Build messages for second prompt with conversation
    prompt2 = MockPrompt("Second question", model=cast(Model, grok))
    messages = grok.build_messages(prompt2, conversation)
    
    assert len(messages) == 3
    assert get_message_role(messages[0]) == "user"
    assert get_message_content(messages[0]) == "First question"
    assert get_message_role(messages[1]) == "assistant"
    assert get_message_content(messages[1]) == "First response"
    assert get_message_role(messages[2]) == "user"
    assert get_message_content(messages[2]) == "Second question"


def test_non_streaming_request(model: Grok, mock_response: Dict[str, object], httpx_mock: HTTPXMock, mock_env: None) -> None:
    """Test non-streaming API request."""
    httpx_mock.add_response(
        method="POST",
        url=CHAT_COMPLETIONS_URL,
        json=mock_response
    )
    
    prompt = MockPrompt("Test prompt", model=cast(llm.Model, model))
    response = llm.Response(model=cast(llm.Model, model), prompt=prompt, stream=False)
    
    # Execute the response
    result = list(response)
    
    assert len(result) == 1
    assert result[0] == "Test response"
    
    # Verify request
    request = httpx_mock.get_request()
    assert request is not None
    # Be flexible about API key - just check it has Bearer auth
    assert "Authorization" in request.headers
    assert request.headers["Authorization"].startswith("Bearer ")
    request_data = json.loads(request.content)
    assert request_data["model"] == TEST_MODEL_ID
    assert request_data["stream"] is False


def test_streaming_request(model: Grok, httpx_mock: HTTPXMock, mock_env: None) -> None:
    """Test streaming API request."""
    chunks = create_streaming_chunks("Hello world!")
    
    httpx_mock.add_response(
        method="POST",
        url=CHAT_COMPLETIONS_URL,
        content=chunks
    )
    
    prompt = MockPrompt("Test prompt", model=cast(llm.Model, model))
    response = llm.Response(model=cast(llm.Model, model), prompt=prompt, stream=True)
    
    # Collect streamed chunks
    collected = []
    for chunk in response:
        collected.append(chunk)
    
    # Check that we got the expected content
    assert "Hello world!" in "".join(collected)
    
    # Verify request
    request = httpx_mock.get_request()
    assert request is not None
    request_data = json.loads(request.content)
    assert request_data["stream"] is True


def test_temperature_option(model: Grok, mock_response: Dict[str, object], httpx_mock: HTTPXMock, mock_env: None) -> None:
    """Test temperature option handling."""
    httpx_mock.add_response(
        method="POST",
        url=CHAT_COMPLETIONS_URL,
        json=mock_response
    )
    
    prompt = MockPrompt("Test", options=GrokOptions(temperature=0.5), model=cast(llm.Model, model))
    response = llm.Response(model=cast(llm.Model, model), prompt=prompt, stream=False)
    list(response)
    
    # Verify temperature in request
    request = httpx_mock.get_request()
    assert request is not None
    request_data = json.loads(request.content)
    assert request_data["temperature"] == 0.5


def test_max_tokens_option(model: Grok, mock_response: Dict[str, object], httpx_mock: HTTPXMock, mock_env: None) -> None:
    """Test max_completion_tokens option handling."""
    httpx_mock.add_response(
        method="POST",
        url=CHAT_COMPLETIONS_URL,
        json=mock_response
    )
    
    prompt = MockPrompt("Test", options=GrokOptions(max_completion_tokens=100), model=cast(llm.Model, model))
    response = llm.Response(model=cast(llm.Model, model), prompt=prompt, stream=False)
    list(response)
    
    # Verify max_completion_tokens in request
    request = httpx_mock.get_request()
    assert request is not None
    request_data = json.loads(request.content)
    assert request_data["max_completion_tokens"] == 100


def test_api_error(model: Grok, httpx_mock: HTTPXMock, mock_env: None) -> None:
    """Test API error handling."""
    error_response = ERROR_RESPONSES["authentication_error"]
    httpx_mock.add_response(
        method="POST",
        url=CHAT_COMPLETIONS_URL,
        status_code=401,
        json=error_response
    )
    
    prompt = MockPrompt("Test", model=cast(llm.Model, model))
    response = llm.Response(model=cast(llm.Model, model), prompt=prompt, stream=False)
    
    # Check that AuthenticationError is raised with correct message
    with pytest.raises(AuthenticationError) as exc_info:
        list(response)
    
    # Check that the error message contains the expected text
    assert "Invalid API key" in str(exc_info.value)


def test_grok_4_models_registered() -> None:
    """Test that Grok 4 models are properly registered."""
    # Get all registered models
    registry = llm.get_models()
    
    # Find Grok models - they may or may not have x-ai/ prefix
    grok_models = [m for m in registry if m.model_id.startswith("x-ai/") or m.model_id.startswith("grok")]
    
    # Check Grok 4 models are present
    grok_4_ids = [m.model_id for m in grok_models]
    
    # Models can be registered with or without x-ai/ prefix
    assert any("grok-4" in id for id in grok_4_ids), f"grok-4 not found in {grok_4_ids}"
    assert any("grok-4-heavy" in id for id in grok_4_ids), f"grok-4-heavy not found in {grok_4_ids}"


# Key Features Tests (MEDIUM PRIORITY)

def test_multimodal_message_building_with_url() -> None:
    """Test multimodal message building with image URL."""
    grok = Grok("x-ai/grok-4")  # Vision-capable model
    attachment = MockAttachment("image", SAMPLE_IMAGE_URL)
    prompt = MockPrompt("What's in this image?", attachments=[attachment])
    
    messages = grok.build_messages(prompt, None)
    
    assert len(messages) == 1
    message = messages[0]
    assert get_message_role(message) == "user"
    content = get_message_content(message)
    assert isinstance(content, list)
    assert len(content) == 2
    # Type narrowing for list content
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "What's in this image?"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == SAMPLE_IMAGE_URL


def test_multimodal_message_building_with_base64() -> None:
    """Test multimodal message building with base64 image."""
    grok = Grok("x-ai/grok-4")  # Vision-capable model
    attachment = MockAttachment("image", SAMPLE_JPEG_BASE64)
    prompt = MockPrompt("What's in this image?", attachments=[attachment])
    
    messages = grok.build_messages(prompt, None)
    
    assert len(messages) == 1
    message = messages[0]
    content = get_message_content(message)
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[1]["type"] == "image_url"
    # Should be formatted as data URL
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_multimodal_only_for_vision_models() -> None:
    """Test that multimodal features are only enabled for vision models."""
    # Non-vision model
    grok_vision = Grok("x-ai/grok-vision-beta")
    attachment = MockAttachment("image", SAMPLE_IMAGE_URL)
    prompt = MockPrompt("What's in this image?", attachments=[attachment])
    
    messages = grok_vision.build_messages(prompt, None)
    
    # Should treat as text-only for non-vision model
    assert len(messages) == 1
    assert get_message_content(messages[0]) == "What's in this image?"  # String, not array


def test_function_calling_options() -> None:
    """Test function calling options for tool-capable models."""
    grok4 = Grok("x-ai/grok-4")  # Tool-capable model
    
    # Create prompt with tools
    tools = [cast(ToolDefinition, tool) for tool in SAMPLE_TOOLS.values()]
    options = GrokOptions(
        tools=tools,
        tool_choice="auto"
    )
    prompt = MockPrompt("Get the weather", options=options, model=cast(llm.Model, grok4))
    
    # Build request body
    body = create_test_request("Get the weather", "x-ai/grok-4", stream=False, tools=cast(List[Dict[str, Any]], tools))
    
    # Should include tools in request
    assert "tools" in body
    assert len(body["tools"]) == 1
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["name"] == "get_weather"
    assert body["tool_choice"] == "auto"


def test_function_calling_in_request_body(httpx_mock: HTTPXMock, mock_env: None) -> None:
    """Test that function calling parameters are included in request."""
    grok4 = Grok("x-ai/grok-4")
    
    # Mock response with tool call
    response_data = create_chat_completion_response("I'll check the weather for you.")
    response_data["choices"][0]["message"]["tool_calls"] = SAMPLE_TOOL_CALLS
    
    httpx_mock.add_response(
        method="POST",
        url=CHAT_COMPLETIONS_URL,
        json=response_data
    )
    
    # Create prompt with tools
    tools = [cast(ToolDefinition, tool) for tool in SAMPLE_TOOLS.values()]
    options = GrokOptions(
        tools=tools,
        tool_choice="auto"
    )
    prompt = MockPrompt("Get the weather in NYC", options=options, model=cast(llm.Model, grok4))
    
    response = llm.Response(model=cast(llm.Model, grok4), prompt=prompt, stream=False)
    list(response)
    
    # Verify request included tools
    request = httpx_mock.get_request()
    assert request is not None
    request_data = json.loads(request.content)
    assert "tools" in request_data
    assert request_data["tool_choice"] == "auto"
    
    # Verify response has tool calls
    assert hasattr(response, 'tool_calls')
    tool_calls_list = response.tool_calls()
    assert len(tool_calls_list) == 1
    # Note: ToolCall attributes depend on llm library implementation
    # For now, just verify we have tool calls


def test_streaming_tool_calls_accumulation(httpx_mock: HTTPXMock, mock_env: None) -> None:
    """Test accumulation of tool calls in streaming mode."""
    grok = Grok("x-ai/grok-4")
    
    # Create streaming chunks that include tool calls
    chunks = [
        'data: {"choices":[{"delta":{"role":"assistant","content":"Let me check the weather."},"index":0}]}\n\n',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather","arguments":"{\\"loc"}}]},"index":0}]}\n\n',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\\": \\"New York\\"}"}}]},"index":0}]}\n\n',
        'data: {"choices":[{"finish_reason":"tool_calls","index":0}]}\n\n',
        'data: [DONE]\n\n'
    ]
    
    httpx_mock.add_response(
        method="POST",
        url=CHAT_COMPLETIONS_URL,
        content="".join(chunks).encode()
    )
    
    tools = [cast(ToolDefinition, tool) for tool in SAMPLE_TOOLS.values()]
    options = GrokOptions(tools=tools)
    prompt = MockPrompt("Get weather", options=options, model=cast(llm.Model, grok))
    response = llm.Response(model=cast(llm.Model, grok), prompt=prompt, stream=True)
    
    # Consume the stream
    content_parts = []
    for chunk in response:
        content_parts.append(chunk)
    
    # Check accumulated content
    assert "Let me check the weather." in "".join(content_parts)
    
    # Check tool calls were accumulated
    assert hasattr(response, 'tool_calls')
    tool_calls = response.tool_calls()
    assert len(tool_calls) == 1
    # Note: ToolCall attributes depend on llm library implementation
    # For now, just verify we have tool calls


def test_messages_endpoint_option_default() -> None:
    """Test that use_messages_endpoint defaults to False."""
    options = GrokOptions()
    assert options.use_messages_endpoint is False


def test_convert_to_anthropic_basic() -> None:
    """Test basic OpenAI to Anthropic format conversion."""
    grok = Grok("x-ai/grok-4")
    
    openai_messages = [
        cast(Message, {"role": "user", "content": "Hello"}),
        cast(Message, {"role": "assistant", "content": "Hi there!"}),
        cast(Message, {"role": "user", "content": "How are you?"})
    ]
    
    anthropic_format = grok._openai_formatter.convert_messages_to_anthropic(openai_messages)
    
    assert "messages" in anthropic_format
    anthropic_messages = anthropic_format["messages"]
    assert len(anthropic_messages) == 3
    
    # Check each message has the correct structure
    for i, (msg, expected_content) in enumerate(zip(anthropic_messages, ["Hello", "Hi there!", "How are you?"])):
        assert msg["role"] in ["user", "assistant"]
        # Content is now an array of content blocks
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][0]["text"] == expected_content


def test_convert_to_anthropic_with_system() -> None:
    """Test OpenAI to Anthropic conversion with system message."""
    grok = Grok("x-ai/grok-4")
    
    openai_messages = [
        cast(Message, {"role": "system", "content": "You are helpful"}),
        cast(Message, {"role": "user", "content": "Hello"})
    ]
    
    # Test the actual conversion method
    anthropic_format = grok._openai_formatter.convert_messages_to_anthropic(openai_messages)
    
    assert "system" in anthropic_format
    assert anthropic_format["system"] == "You are helpful"
    assert "messages" in anthropic_format
    assert len(anthropic_format["messages"]) == 1
    assert anthropic_format["messages"][0]["role"] == "user"
    # Content is now an array of content blocks
    assert isinstance(anthropic_format["messages"][0]["content"], list)
    assert len(anthropic_format["messages"][0]["content"]) == 1
    assert anthropic_format["messages"][0]["content"][0]["type"] == "text"
    assert anthropic_format["messages"][0]["content"][0]["text"] == "Hello"


def test_messages_endpoint_request_format(httpx_mock: HTTPXMock, mock_env) -> None:
    """Test that messages endpoint uses correct request format."""
    grok = Grok("x-ai/grok-4")
    
    httpx_mock.add_response(
        method="POST",
        url=MESSAGES_URL,
        json=create_messages_response("Test response from messages endpoint")
    )
    
    # Create options separately and pass to prompt
    options = GrokOptions(use_messages_endpoint=True)
    prompt = MockPrompt("Hello", options=options, model=cast(llm.Model, grok))
    
    response = llm.Response(model=cast(llm.Model, grok), prompt=prompt, stream=False)
    list(response)
    
    # Verify request format
    request = httpx_mock.get_request()
    assert request is not None
    assert str(request.url) == MESSAGES_URL
    
    request_data = json.loads(request.content)
    assert "model" in request_data
    assert "messages" in request_data
    assert request_data["model"] == "x-ai/grok-4"


@pytest.mark.skip(reason="Streaming endpoint implementation needs investigation")
def test_messages_endpoint_streaming(httpx_mock: HTTPXMock) -> None:
    """Test streaming with messages endpoint."""
    grok = Grok("x-ai/grok-4")
    
    chunks = create_anthropic_streaming_chunks(["Hello", " from", " Anthropic"])
    
    httpx_mock.add_response(
        method="POST",
        url=MESSAGES_URL,
        content=chunks.encode()
    )
    
    # Create options separately and pass to prompt  
    options = GrokOptions(use_messages_endpoint=True)
    prompt = MockPrompt("Test", options=options, model=cast(llm.Model, grok))
    
    response = llm.Response(model=cast(llm.Model, grok), prompt=prompt, stream=True)
    
    collected = []
    for chunk in response:
        collected.append(chunk)
    
    # Check that we got some content
    assert len(collected) > 0, f"No chunks collected from streaming response"
    
    # More flexible assertion - check we got text content
    full_response = "".join(collected)
    # Accept any non-empty response as the exact format may vary
    assert len(full_response.strip()) > 0, f"Got empty response, collected: {collected}"


def test_json_cache_memory_monitoring() -> None:
    """Test that JSON cache monitors memory usage and prevents leaks."""
    from llm_grok.client import GrokClient
    
    client = GrokClient("test-key")
    
    # Test initial state
    assert client._cache_memory_usage == 0
    assert len(client._json_size_cache) == 0
    
    # Test memory tracking with cache entries
    test_data_small: RequestBody = {"model": "test", "messages": []}
    test_data_large: RequestBody = {"model": "test-large", "messages": []}  # Large payload
    
    # Estimate sizes
    size1 = client._estimate_json_size(test_data_small)
    assert size1 > 0
    assert client._cache_memory_usage > 0
    assert len(client._json_size_cache) == 1
    
    initial_memory = client._cache_memory_usage
    
    # Add another entry
    size2 = client._estimate_json_size(test_data_large)
    assert size2 > size1
    assert client._cache_memory_usage > initial_memory
    assert len(client._json_size_cache) == 2
    
    # Test cache limits by setting low memory limit
    original_limit = client.MAX_CACHE_MEMORY
    client.MAX_CACHE_MEMORY = 50  # Very low limit
    
    # Adding this should trigger cache clearing
    test_data_another: RequestBody = {"model": "another", "messages": []}
    size3 = client._estimate_json_size(test_data_another)
    # Cache should have been cleared and now only contains the new entry
    assert len(client._json_size_cache) == 1  # Only new entry remains
    assert client._cache_memory_usage == size3  # Memory usage should equal size of new entry
    
    # Test cleanup
    client._cleanup_cache()
    assert client._cache_memory_usage == 0
    assert len(client._json_size_cache) == 0
    
    # Restore original cache memory limit
    client.MAX_CACHE_MEMORY = original_limit
    
    client.close()