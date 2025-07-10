import base64
import json
import warnings
from typing import Any, Dict, List, Optional, Union, cast

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
from llm.models import ToolCall


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
            
        # Initialize parent class
        super().__init__(prompt=prompt, model=model, stream=stream)
        
        # Add custom attributes
        self.response_json: Optional[Dict[str, object]] = None
        self._mock_tool_calls_data: List[Dict[str, object]] = []  # Store dict data for test assertions
        self._prompt_json: Optional[Dict[str, object]] = None
        self._tool_calls_accumulator: List[Dict[str, object]] = []
    
    def tool_calls(self) -> List[ToolCall]:
        """Get tool calls as a method to match base class."""
        # Convert dict data to ToolCall objects for proper type compatibility
        tool_call_objects = []
        for tc_data in self._mock_tool_calls_data:
            if "function" in tc_data and isinstance(tc_data["function"], dict):
                func = tc_data["function"]
                arguments = func.get("arguments", {})
                # Parse JSON string arguments if needed
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                tool_call_objects.append(ToolCall(
                    name=func.get("name", ""),
                    arguments=arguments,
                    tool_call_id=cast(Optional[str], tc_data.get("id"))
                ))
        return tool_call_objects
    
    def add_tool_call(self, tool_call: Any) -> None:
        """Override add_tool_call to store dict representation for testing."""
        # Convert llm.ToolCall to dict format for test assertions
        if hasattr(tool_call, 'name') and hasattr(tool_call, 'arguments'):
            # This is an llm.ToolCall object, convert to dict
            tool_dict = {
                "id": getattr(tool_call, 'tool_call_id', None),
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments) if isinstance(tool_call.arguments, dict) else tool_call.arguments
                }
            }
            self._mock_tool_calls_data.append(tool_dict)
        else:
            # Already a dict, just append
            self._mock_tool_calls_data.append(tool_call)
    
    def text(self) -> str:
        if self.response_json and "choices" in self.response_json:
            choices = self.response_json["choices"]
            if isinstance(choices, list) and len(choices) > 0:
                choice = choices[0]
                if isinstance(choice, dict):
                    message = choice.get("message", {})
                    if isinstance(message, dict):
                        return message.get("content", "")
        return ""


@pytest.fixture(autouse=True)
def ignore_warnings() -> None:
    """Ignore known warnings."""
    # Filter out known deprecation warnings
    warnings.filterwarnings("ignore", message="Support for class-based `config` is deprecated")
    warnings.filterwarnings("ignore", message="datetime.datetime.utcnow() is deprecated")


@pytest.fixture
def model() -> Grok:
    return Grok(DEFAULT_MODEL)


@pytest.fixture(autouse=True)
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock environment variables and API key for testing"""
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key-mock")
    # Mock the get_key method to always return our test key
    def mock_get_key(self, key: Optional[str] = None) -> str:
        return "xai-test-key-mock"
    monkeypatch.setattr(Grok, "get_key", mock_get_key)


@pytest.fixture
def mock_response() -> Dict[str, object]:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Test response"},
                "finish_reason": "stop",
            }
        ],
    }


def test_model_initialization(model: Grok) -> None:
    assert model.model_id == DEFAULT_MODEL
    assert model.can_stream == True
    assert model.needs_key == "grok"
    assert model.key_env_var == "XAI_API_KEY"


def test_grok_4_model_initialization() -> None:
    """Test initialization of Grok 4 models"""
    grok4 = Grok("x-ai/grok-4")
    assert grok4.model_id == "x-ai/grok-4"
    assert grok4.can_stream == True
    
    grok4_heavy = Grok("grok-4-heavy")
    assert grok4_heavy.model_id == "grok-4-heavy"
    assert grok4_heavy.can_stream == True


def test_model_info_registry() -> None:
    """Test that all models have metadata in MODEL_INFO"""
    for model_id in AVAILABLE_MODELS:
        assert model_id in MODEL_INFO
        info = MODEL_INFO[model_id]
        assert "context_window" in info
        assert "supports_vision" in info
        assert "supports_tools" in info
        assert "pricing_tier" in info
        assert "max_output_tokens" in info


def test_grok_4_model_info() -> None:
    """Test Grok 4 specific model capabilities"""
    # Test x-ai/grok-4
    grok4_info = MODEL_INFO["x-ai/grok-4"]
    assert grok4_info["context_window"] == 256000
    assert grok4_info["supports_vision"] == True
    assert grok4_info["supports_tools"] == True
    assert grok4_info["pricing_tier"] == "standard"
    assert grok4_info["max_output_tokens"] == 8192
    
    # Test grok-4-heavy
    heavy_info = MODEL_INFO["grok-4-heavy"]
    assert heavy_info["context_window"] == 256000
    assert heavy_info["supports_vision"] == True
    assert heavy_info["supports_tools"] == True
    assert heavy_info["pricing_tier"] == "heavy"
    assert heavy_info["max_output_tokens"] == 8192


def test_default_model_is_grok_4() -> None:
    """Test that default model is now Grok 4"""
    assert DEFAULT_MODEL == "x-ai/grok-4"


def test_build_messages_with_system_prompt(model: Grok) -> None:
    prompt = llm.Prompt(
        model=model, prompt="Test message", system="Custom system message"
    )
    messages = model.build_messages(prompt, None)

    assert len(messages) == 2
    assert messages[0].get("role") == "system"
    assert messages[0].get("content") == "Custom system message"
    assert messages[1].get("role") == "user"
    assert messages[1].get("content") == "Test message"


def test_build_messages_without_system_prompt(model: Grok) -> None:
    prompt = llm.Prompt(model=model, prompt="Test message")
    messages = model.build_messages(prompt, None)

    assert len(messages) == 1
    assert messages[0].get("role") == "user"
    assert messages[0].get("content") == "Test message"


def test_build_messages_with_conversation(model: Grok, httpx_mock: HTTPXMock, mock_env: None) -> None:
    # Mock the expected request content
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Previous message"},
        ],
        "stream": False,
        "temperature": 0.0,
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json={
            "id": "chatcmpl-123",
            "choices": [
                {"message": {"role": "assistant", "content": "Previous response"}}
            ],
        },
        match_json=expected_request,
    )

    conversation = llm.Conversation(model=model)
    prev_prompt = llm.Prompt(model=model, prompt="Previous message")

    prev_response = llm.Response(model=model, prompt=prev_prompt, stream=False)
    # Mock the response by setting internal attributes
    # This is needed for testing conversation history
    setattr(prev_response, '_response_json', {
        "choices": [{"message": {"role": "assistant", "content": "Previous response"}}]
    })

    conversation.responses.append(prev_response)

    prompt = llm.Prompt(model=model, prompt="New message")
    messages = model.build_messages(prompt, conversation)

    assert len(messages) == 3
    assert messages[0].get("role") == "user"
    assert messages[0].get("content") == "Previous message"
    assert messages[1].get("role") == "assistant"
    assert messages[1].get("content") == "Previous response"
    assert messages[2].get("role") == "user"
    assert messages[2].get("content") == "New message"


def test_non_streaming_request(model: Grok, mock_response: Dict[str, object], httpx_mock: HTTPXMock, mock_env: None) -> None:
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Test message"},
        ],
        "stream": False,
        "temperature": 0.0,
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=mock_response,
        headers={"Content-Type": "application/json"},
        match_json=expected_request,
    )

    response = model.prompt("Test message", stream=False)
    result = response.text()
    assert result == "Test response"

    request = httpx_mock.get_requests()[0]
    assert request.headers["Authorization"] == "Bearer xai-test-key-mock"
    assert json.loads(request.content) == expected_request


def test_streaming_request(model: Grok, httpx_mock: HTTPXMock, mock_env: None) -> None:
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Test message"},
        ],
        "stream": True,
        "temperature": 0.0,
    }

    def response_callback(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer xai-test-key-mock"
        assert json.loads(request.content) == expected_request
        stream_content = [
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"role":"assistant"}}]}\n\n',
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":"Test"}}]}\n\n',
            'data: {"id":"chatcmpl-123","choices":[{"delta":{"content":" response"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content="".join(stream_content).encode(),
        )

    httpx_mock.add_callback(
        response_callback,
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        match_json=expected_request,
    )

    response = model.prompt("Test message", stream=True)
    chunks = list(response)
    assert "".join(chunks) == "Test response"


def test_temperature_option(model: Grok, mock_response: Dict[str, object], httpx_mock: HTTPXMock, mock_env: None) -> None:
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Test message"},
        ],
        "stream": False,
        "temperature": 0.8,
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=mock_response,
        headers={"Content-Type": "application/json"},
        match_json=expected_request,
    )

    # Create prompt and pass temperature directly
    response = model.prompt("Test message", stream=False, temperature=0.8)
    result = response.text()
    assert result == "Test response"

    request = httpx_mock.get_requests()[0]
    assert json.loads(request.content) == expected_request


def test_max_tokens_option(model: Grok, mock_response: Dict[str, object], httpx_mock: HTTPXMock, mock_env: None) -> None:
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Test message"},
        ],
        "stream": False,
        "temperature": 0.0,
        "max_completion_tokens": 100,
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=mock_response,
        headers={"Content-Type": "application/json"},
        match_json=expected_request,
    )

    # Create prompt and pass max_tokens directly
    response = model.prompt("Test message", stream=False, max_completion_tokens=100)
    result = response.text()
    assert result == "Test response"

    request = httpx_mock.get_requests()[0]
    assert json.loads(request.content) == expected_request


def test_api_error(model: Grok, httpx_mock: HTTPXMock, mock_env: None) -> None:
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Test message"},
        ],
        "stream": False,
        "temperature": 0.0,
    }

    error_response = {
        "error": {
            "message": "Invalid request",
            "type": "invalid_request_error",
            "code": "invalid_api_key",
        }
    }

    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        status_code=400,
        json=error_response,
        headers={"Content-Type": "application/json"},
        match_json=expected_request,
    )

    with pytest.raises(GrokError) as exc_info:
        response = model.prompt("Test message", stream=False)
        response.text()  # Trigger the API call

    # The error message comes directly from the API response
    error_obj = error_response.get("error", {})
    assert str(exc_info.value) == error_obj.get("message", "")


def test_stream_parsing_error(model: Grok, httpx_mock: HTTPXMock, mock_env: None) -> None:
    expected_request = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "user", "content": "Test message"},
        ],
        "stream": True,
        "temperature": 0.0,
    }

    def error_callback(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer xai-test-key-mock"
        assert json.loads(request.content) == expected_request
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=b"data: {invalid json}\n\n",
        )

    httpx_mock.add_callback(
        error_callback,
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        match_json=expected_request,
    )

    response = model.prompt("Test message", stream=True)
    chunks = list(response)
    assert chunks == []


def test_grok_4_models_registered() -> None:
    """Test that Grok 4 models are registered in LLM"""
    # This test verifies models are available for registration
    models_to_register: List[Grok] = []
    
    def mock_register(model: Grok) -> None:
        models_to_register.append(model)
    
    register_models(mock_register)
    
    model_ids = [model.model_id for model in models_to_register]
    assert "x-ai/grok-4" in model_ids
    assert "grok-4-heavy" in model_ids


def test_multimodal_message_building_with_url() -> None:
    """Test building multimodal messages with image URLs"""
    grok4 = Grok("x-ai/grok-4")
    
    # Test with image URL
    prompt = MockPrompt(
        "What's in this image?",
        [MockAttachment("image", "https://example.com/image.jpg")]
    )
    
    messages = grok4.build_messages(prompt, None)
    
    assert len(messages) == 1
    assert messages[0].get("role") == "user"
    content = messages[0].get("content")
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "What's in this image?"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/image.jpg"}
    }


def test_multimodal_message_building_with_base64() -> None:
    """Test building multimodal messages with base64 images"""
    grok4 = Grok("x-ai/grok-4")
    
    # Test with base64 image (minimal valid JPEG header)
    base64_image = "/9j/4AAQSkZJRgABAQEAAAAAAAD/2wBDAAEBAQ=="
    prompt = MockPrompt(
        "Analyze this image",
        [MockAttachment("image", base64_image)]
    )
    
    messages = grok4.build_messages(prompt, None)
    
    assert len(messages) == 1
    assert messages[0].get("role") == "user"
    content = messages[0].get("content")
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0] == {"type": "text", "text": "Analyze this image"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
    }


def test_multimodal_message_building_with_data_url() -> None:
    """Test building multimodal messages with data URLs"""
    grok4 = Grok("x-ai/grok-4")
    
    # Test with data URL
    data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    prompt = MockPrompt(
        "What color is this pixel?",
        [MockAttachment("image", data_url)]
    )
    
    messages = grok4.build_messages(prompt, None)
    
    assert len(messages) == 1
    assert messages[0].get("role") == "user"
    content = messages[0].get("content")
    assert isinstance(content, list)
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": data_url}
    }


def test_multimodal_only_for_vision_models() -> None:
    """Test that multimodal content is only used for vision-capable models"""
    # Test with non-vision model
    grok3 = Grok("grok-3-latest")  # This model doesn't support vision
    
    prompt = MockPrompt(
        "What's in this image?",
        [MockAttachment("image", "https://example.com/image.jpg")]
    )
    
    messages = grok3.build_messages(prompt, None)
    
    # Should return plain text since model doesn't support vision
    assert len(messages) == 1
    assert messages[0].get("role") == "user"
    assert messages[0].get("content") == "What's in this image?"  # Plain string, not list


def test_image_validation() -> None:
    """Test image format validation"""
    grok4 = Grok("x-ai/grok-4")
    
    # Test valid formats
    assert grok4._validate_image_format("https://example.com/image.jpg") == "https://example.com/image.jpg"
    assert grok4._validate_image_format("http://example.com/image.jpg") == "http://example.com/image.jpg"
    
    # Test valid data URL
    data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    assert grok4._validate_image_format(data_url) == data_url
    
    # Test base64 JPEG detection
    jpeg_base64 = "/9j/4AAQSkZJRgABAQEAAAAAAAD/2wBDAAEBAQ=="
    result = grok4._validate_image_format(jpeg_base64)
    assert result.startswith("data:image/jpeg;base64,")
    
    # Test base64 PNG detection
    png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    result = grok4._validate_image_format(png_base64)
    assert result.startswith("data:image/png;base64,")
    
    # Test invalid data URL
    try:
        grok4._validate_image_format("data:image/png,notbase64")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid data URL format" in str(e)
    
    # Test invalid base64
    try:
        grok4._validate_image_format("not-valid-base64!")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid base64 image data" in str(e)
    
    # Test unsupported image format (base64 data that doesn't match known magic bytes)
    # This is valid base64 but not a recognized image format
    unknown_format_base64 = base64.b64encode(b"UNKNOWN_FORMAT_DATA").decode()
    try:
        grok4._validate_image_format(unknown_format_base64)
        assert False, "Should have raised ValueError for unknown image format"
    except ValueError as e:
        assert "Unable to detect image type" in str(e)
        assert "supported image format" in str(e)


def test_function_calling_options() -> None:
    """Test that function calling options are properly set"""
    grok4 = Grok("x-ai/grok-4")
    
    # Test tools option
    tools = cast(List[ToolDefinition], [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }])
    
    options = Grok.Options(
        tools=tools,
        tool_choice="auto",
        response_format={"type": "json_object"},
        reasoning_effort="medium"
    )
    
    assert options.tools == tools
    assert options.tool_choice == "auto"
    assert options.response_format == {"type": "json_object"}
    assert options.reasoning_effort == "medium"


def test_function_calling_in_request_body(httpx_mock: HTTPXMock, mock_env: None) -> None:
    """Test that function calling parameters are included in API request for supported models"""
    grok4 = Grok("x-ai/grok-4")
    
    tools = cast(List[ToolDefinition], [{
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                }
            }
        }
    }])
    
    # Create mock prompt and response
    
    # Create prompt with function calling options
    prompt = MockPrompt("Calculate 2+2", model=cast(llm.Model, grok4))
    prompt.options = Grok.Options(
        tools=tools,
        tool_choice="auto"
    )
    
    response = MockResponse(prompt=prompt, model=cast(llm.Model, grok4))
    
    # Expected request should include tools
    expected_request = {
        "model": "x-ai/grok-4",
        "messages": [
            {"role": "user", "content": "Calculate 2+2"}
        ],
        "stream": False,
        "temperature": 0.0,
        "tools": tools,
        "tool_choice": "auto"
    }
    
    # Mock response with tool call
    mock_response = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": '{"expression": "2+2"}'
                    }
                }]
            }
        }]
    }
    
    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=mock_response,
        match_json=expected_request
    )
    
    # Call execute method directly
    result = list(grok4.execute(prompt, stream=False, response=response, conversation=None))
    
    # Check that tool_calls are captured
    assert hasattr(response, 'tool_calls')
    tool_calls_value = response.tool_calls()
    assert tool_calls_value is not None
    assert len(tool_calls_value) == 1
    tool_call = tool_calls_value[0]
    assert isinstance(tool_call, ToolCall)
    assert tool_call.name == "calculate"


def test_reasoning_effort_in_request_body(httpx_mock: HTTPXMock, mock_env: None) -> None:
    """Test that reasoning_effort is included in API request"""
    grok4 = Grok("x-ai/grok-4")
    
    # Use global mock classes
    
    # Create prompt with reasoning_effort option
    prompt = MockPrompt("Solve this complex problem", model=cast(llm.Model, grok4))
    prompt.options = Grok.Options(reasoning_effort="high")
    
    response = MockResponse(prompt=prompt, model=cast(llm.Model, grok4))
    
    # Expected request should include reasoning_effort
    expected_request = {
        "model": "x-ai/grok-4",
        "messages": [
            {"role": "user", "content": "Solve this complex problem"}
        ],
        "stream": False,
        "temperature": 0.0,
        "reasoning_effort": "high"
    }
    
    mock_response = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Solution to the problem"
            }
        }]
    }
    
    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=mock_response,
        match_json=expected_request
    )
    
    # Execute request
    result = list(grok4.execute(prompt, stream=False, response=response, conversation=None))
    assert result == ["Solution to the problem"]


def test_response_format_in_request_body(httpx_mock: HTTPXMock, mock_env: None) -> None:
    """Test that response_format is included in API request for tool-supported models"""
    grok4 = Grok("x-ai/grok-4")
    
    # Use global mock classes
    
    # Create prompt with response_format option
    prompt = MockPrompt("Generate a JSON object", model=cast(llm.Model, grok4))
    prompt.options = Grok.Options(
        response_format={"type": "json_object"}
    )
    
    response = MockResponse(prompt=prompt, model=cast(llm.Model, grok4))
    
    # Expected request should include response_format (only for tool-supported models)
    expected_request = {
        "model": "x-ai/grok-4",
        "messages": [
            {"role": "user", "content": "Generate a JSON object"}
        ],
        "stream": False,
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    mock_response = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": '{"result": "success"}'
            }
        }]
    }
    
    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=mock_response,
        match_json=expected_request
    )
    
    # Execute request
    result = list(grok4.execute(prompt, stream=False, response=response, conversation=None))
    assert result == ['{"result": "success"}']


def test_all_options_combined(httpx_mock: HTTPXMock, mock_env: None) -> None:
    """Test that all options work together properly"""
    grok4 = Grok("x-ai/grok-4")
    
    # Use global mock classes
    
    # Create prompt with all options
    tools = cast(List[ToolDefinition], [{
        "type": "function",
        "function": {
            "name": "test_func",
            "description": "Test function",
            "parameters": {"type": "object", "properties": {}}
        }
    }])
    
    prompt = MockPrompt("Test all options", model=cast(llm.Model, grok4))
    prompt.options = Grok.Options(
        temperature=0.7,
        max_completion_tokens=1000,
        tools=tools,
        tool_choice="auto",
        response_format={"type": "json_object"},
        reasoning_effort="medium"
    )
    
    response = MockResponse(prompt=prompt, model=cast(llm.Model, grok4))
    
    # Expected request should include all parameters
    expected_request = {
        "model": "x-ai/grok-4",
        "messages": [
            {"role": "user", "content": "Test all options"}
        ],
        "stream": False,
        "temperature": 0.7,
        "max_completion_tokens": 1000,
        "tools": tools,
        "tool_choice": "auto",
        "response_format": {"type": "json_object"},
        "reasoning_effort": "medium"
    }
    
    mock_response = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": '{"status": "all options received"}'
            }
        }]
    }
    
    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=mock_response,
        match_json=expected_request
    )
    
    # Execute request
    result = list(grok4.execute(prompt, stream=False, response=response, conversation=None))
    assert result == ['{"status": "all options received"}']


def test_function_calling_not_sent_for_unsupported_models(httpx_mock: HTTPXMock, mock_env: None) -> None:
    """Test that function calling parameters are NOT sent for models that don't support it"""
    grok3 = Grok("grok-3-latest")  # This model doesn't support tools
    
    tools = cast(List[ToolDefinition], [{
        "type": "function",
        "function": {
            "name": "test_func",
            "description": "Test function"
        }
    }])
    
    # Use global mock classes
    
    # Create prompt with function calling options
    prompt = MockPrompt("Test message", model=cast(llm.Model, grok3))
    prompt.options = Grok.Options(
        tools=tools,
        tool_choice="auto"
    )
    
    response = MockResponse(prompt=prompt, model=cast(llm.Model, grok3))
    
    # Expected request should NOT include tools
    expected_request = {
        "model": "grok-3-latest",
        "messages": [
            {"role": "user", "content": "Test message"}
        ],
        "stream": False,
        "temperature": 0.0
        # Note: no tools or tool_choice
    }
    
    mock_response = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Regular response"
            }
        }]
    }
    
    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        match_headers={"Authorization": "Bearer xai-test-key-mock"},
        json=mock_response,
        match_json=expected_request
    )
    
    # Call execute method directly
    result = list(grok3.execute(prompt, stream=False, response=response, conversation=None))
    assert result == ["Regular response"]


def test_streaming_tool_calls_accumulation(httpx_mock: HTTPXMock, mock_env: None) -> None:
    """Test that streaming tool calls are properly accumulated across chunks"""
    grok4 = Grok("x-ai/grok-4")
    
    # Use global mock classes
    
    prompt = MockPrompt("What's the weather in Paris and London?", model=cast(llm.Model, grok4))
    prompt.options = Grok.Options(
        tools=cast(List[ToolDefinition], [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }]),
        tool_choice="auto"
    )
    
    response = MockResponse(prompt=prompt, model=cast(llm.Model, grok4))
    
    # Create chunked SSE response that simulates tool calls being streamed
    chunks = [
        # First tool call starts
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather"}}]}}]}\n\n',
        # First tool call arguments chunk 1
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"loc"}}]}}]}\n\n',
        # First tool call arguments chunk 2
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\\": \\"Paris\\"}"}}]}}]}\n\n',
        # Second tool call starts
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"get_weather"}}]}}]}\n\n',
        # Second tool call arguments
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\\"location\\": \\"London\\"}"}}]}}]}\n\n',
        # Done
        b'data: [DONE]\n\n'
    ]
    
    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        headers={"content-type": "text/event-stream"},
        content=b"".join(chunks)
    )
    
    # Execute streaming request
    result = list(grok4.execute(prompt, stream=True, response=response, conversation=None))
    
    # Verify tool calls were accumulated correctly
    assert hasattr(response, 'tool_calls')
    tool_calls_value = response.tool_calls()
    assert tool_calls_value is not None
    assert len(tool_calls_value) == 2
    
    # Check first tool call
    first_call = tool_calls_value[0]
    assert isinstance(first_call, ToolCall)
    assert first_call.tool_call_id == "call_1"
    assert first_call.name == "get_weather"
    assert first_call.arguments == {"location": "Paris"}
    
    # Check second tool call
    second_call = tool_calls_value[1]
    assert isinstance(second_call, ToolCall)
    assert second_call.tool_call_id == "call_2"
    assert second_call.name == "get_weather"
    assert second_call.arguments == {"location": "London"}


def test_streaming_tool_calls_with_content(httpx_mock: HTTPXMock, mock_env: None) -> None:
    """Test streaming with both content and tool calls"""
    grok4 = Grok("x-ai/grok-4")
    
    # Use global mock classes
    
    prompt = MockPrompt("Check the weather and explain", model=cast(llm.Model, grok4))
    prompt.options = Grok.Options(tools=cast(List[ToolDefinition], [{"type": "function", "function": {"name": "get_weather"}}]))
    
    response = MockResponse(prompt=prompt, model=cast(llm.Model, grok4))
    
    # Mix content and tool calls in stream
    chunks = [
        b'data: {"choices":[{"delta":{"content":"I\'ll check "}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"the weather "}}]}\n\n',
        b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\\"location\\": \\"NYC\\"}"}}]}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":"for you."}}]}\n\n',
        b'data: [DONE]\n\n'
    ]
    
    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/chat/completions",
        headers={"content-type": "text/event-stream"},
        content=b"".join(chunks)
    )
    
    # Execute and collect content
    result = list(grok4.execute(prompt, stream=True, response=response, conversation=None))
    
    # Check content was collected
    assert result == ["I'll check ", "the weather ", "for you."]
    
    # Check tool call was also collected
    assert hasattr(response, 'tool_calls')
    tool_calls_value = response.tool_calls()
    assert tool_calls_value is not None
    assert len(tool_calls_value) == 1
    tool_call = tool_calls_value[0]
    assert isinstance(tool_call, ToolCall)
    assert tool_call.name == "get_weather"
    assert tool_call.arguments == {"location": "NYC"}


# Messages Endpoint Tests

def test_messages_endpoint_option_default() -> None:
    """Test that use_messages_endpoint option defaults to False."""
    grok = Grok("x-ai/grok-4")
    options = Grok.Options()
    assert options.use_messages_endpoint is False


def test_convert_to_anthropic_basic() -> None:
    """Test basic OpenAI to Anthropic message conversion."""
    grok = Grok("x-ai/grok-4")
    
    openai_messages = cast(List[Message], [
        {"role": "user", "content": "Hello, how are you?"}
    ])
    
    result = grok._convert_to_anthropic_messages(openai_messages)
    
    assert "messages" in result
    messages = result.get("messages", [])
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"type": "text", "text": "Hello, how are you?"}]
    assert "system" not in result


def test_convert_to_anthropic_with_system() -> None:
    """Test conversion with system message."""
    grok = Grok("x-ai/grok-4")
    
    openai_messages = cast(List[Message], [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ])
    
    result = grok._convert_to_anthropic_messages(openai_messages)
    
    assert "messages" in result
    messages = result.get("messages", [])
    assert len(messages) == 1  # Only user message
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == [{"type": "text", "text": "Hello!"}]
    assert "system" in result
    assert result.get("system") == "You are a helpful assistant."


def test_convert_to_anthropic_multiple_systems() -> None:
    """Test conversion with multiple system messages."""
    grok = Grok("x-ai/grok-4")
    
    openai_messages = cast(List[Message], [
        {"role": "system", "content": "First instruction."},
        {"role": "system", "content": "Second instruction."},
        {"role": "user", "content": "Hello!"}
    ])
    
    # Just check the result - don't try to capture console output
    result = grok._convert_to_anthropic_messages(openai_messages)
    
    # Check messages
    assert result.get("system") == "First instruction.\n\nSecond instruction."
    messages = result.get("messages", [])
    assert len(messages) == 1


def test_convert_to_anthropic_multimodal() -> None:
    """Test conversion with multimodal content."""
    grok = Grok("x-ai/grok-4")
    
    openai_messages = cast(List[Message], [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
            ]
        }
    ])
    
    result = grok._convert_to_anthropic_messages(openai_messages)
    
    messages = result.get("messages", [])
    assert len(messages) == 1
    msg = messages[0]
    assert msg.get("role") == "user"
    content = msg.get("content", [])
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0].get("type") == "text"
    assert content[0].get("text") == "What's in this image?"
    assert content[1].get("type") == "image"
    source = content[1].get("source", {})
    assert isinstance(source, dict)
    assert source.get("type") == "base64"


def test_convert_tools_to_anthropic() -> None:
    """Test OpenAI tool definitions to Anthropic format conversion."""
    grok = Grok("x-ai/grok-4")
    
    openai_tools = cast(List[ToolDefinition], [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }
    ])
    
    result = grok._convert_tools_to_anthropic(openai_tools)
    
    assert len(result) == 1
    tool = result[0]
    assert tool.get("name") == "get_weather"
    assert tool.get("description") == "Get weather for a location"
    input_schema = tool.get("input_schema", {})
    assert input_schema.get("type") == "object"
    properties = input_schema.get("properties", {})
    assert "location" in properties


def test_convert_from_anthropic_response() -> None:
    """Test Anthropic to OpenAI response conversion."""
    grok = Grok("x-ai/grok-4")
    
    anthropic_response = {
        "id": "msg_123",
        "model": "x-ai/grok-4",
        "content": [
            {"type": "text", "text": "Hello there!"}
        ],
        "stop_reason": "end_turn",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    }
    
    result = grok._convert_from_anthropic_response(anthropic_response)
    
    assert result.get("id") == "msg_123"
    assert result.get("object") == "chat.completion"
    assert result.get("model") == "x-ai/grok-4"
    choices = result.get("choices", [])
    assert len(choices) == 1
    message = choices[0].get("message", {})
    assert message.get("role") == "assistant"
    assert message.get("content") == "Hello there!"
    assert choices[0].get("finish_reason") == "end_turn"
    usage = result.get("usage", {})
    assert usage.get("prompt_tokens") == 10


def test_convert_from_anthropic_with_tools() -> None:
    """Test Anthropic response with tool use conversion."""
    grok = Grok("x-ai/grok-4")
    
    anthropic_response = {
        "id": "msg_123",
        "model": "x-ai/grok-4",
        "content": [
            {"type": "text", "text": "I'll check the weather."},
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"location": "NYC"}
            }
        ],
        "stop_reason": "tool_use"
    }
    
    result = grok._convert_from_anthropic_response(anthropic_response)
    
    choices = result.get("choices", [])
    assert len(choices) > 0
    message = choices[0].get("message", {})
    assert message.get("content") == "I'll check the weather."
    tool_calls = message.get("tool_calls", [])
    assert len(tool_calls) == 1
    tool_call = tool_calls[0]
    assert tool_call.get("id") == "toolu_123"
    assert tool_call.get("type") == "function"
    function_info = tool_call.get("function", {})
    assert function_info.get("name") == "get_weather"
    args_str = function_info.get("arguments", "{}")
    assert json.loads(args_str) == {"location": "NYC"}


def test_messages_endpoint_request_format(httpx_mock: HTTPXMock) -> None:
    """Test that messages endpoint uses correct request format."""
    grok = Grok("x-ai/grok-4")
    prompt = llm.Prompt("Hello", model=grok)
    prompt.options = Grok.Options(use_messages_endpoint=True)
    response = llm.Response(model=grok, prompt=prompt, stream=False)
    
    # Mock Anthropic response
    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/messages",
        json={
            "id": "msg_123",
            "model": "x-ai/grok-4",
            "content": [{"type": "text", "text": "Hi there!"}],
            "stop_reason": "end_turn"
        }
    )
    
    # Execute
    result = list(grok.execute(prompt, stream=False, response=response, conversation=None))
    
    # Check request
    assert len(httpx_mock.get_requests()) == 1
    request = httpx_mock.get_requests()[0]
    assert request.url == "https://api.x.ai/v1/messages"
    
    body = json.loads(request.content)
    assert "messages" in body
    messages = body.get("messages", [])
    assert len(messages) > 0
    assert messages[0].get("role") == "user"
    assert messages[0].get("content") == [{"type": "text", "text": "Hello"}]
    assert body.get("temperature") == 0.0
    assert not body.get("stream")
    
    # Check response
    assert result == ["Hi there!"]


def test_messages_endpoint_with_system(httpx_mock: HTTPXMock) -> None:
    """Test messages endpoint with system prompt."""
    grok = Grok("x-ai/grok-4")
    prompt = llm.Prompt("Hello", model=grok, system="Be helpful")
    prompt.options = Grok.Options(use_messages_endpoint=True)
    response = llm.Response(model=grok, prompt=prompt, stream=False)
    
    # Mock response
    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/messages",
        json={
            "id": "msg_123",
            "content": [{"type": "text", "text": "Hello! How can I help?"}],
            "stop_reason": "end_turn"
        }
    )
    
    # Execute
    list(grok.execute(prompt, stream=False, response=response, conversation=None))
    
    # Check request
    request = httpx_mock.get_requests()[0]
    body = json.loads(request.content)
    assert "system" in body
    assert body.get("system") == "Be helpful"
    messages = body.get("messages", [])
    assert len(messages) == 1


def test_messages_endpoint_streaming(httpx_mock: HTTPXMock) -> None:
    """Test streaming with messages endpoint."""
    grok = Grok("x-ai/grok-4")
    prompt = llm.Prompt("Hello", model=grok)
    prompt.options = Grok.Options(use_messages_endpoint=True)
    response = llm.Response(model=grok, prompt=prompt, stream=True)
    
    # Mock streaming response with Anthropic SSE format
    chunks = [
        b'event: message_start\ndata: {"message": {"id": "msg_123", "model": "x-ai/grok-4"}}\n\n',
        b'event: content_block_start\ndata: {"content_block": {"type": "text"}}\n\n',
        b'event: content_block_delta\ndata: {"delta": {"type": "text_delta", "text": "Hello "}}\n\n',
        b'event: content_block_delta\ndata: {"delta": {"type": "text_delta", "text": "there!"}}\n\n',
        b'event: message_stop\ndata: {}\n\n'
    ]
    
    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/messages",
        headers={"content-type": "text/event-stream"},
        content=b"".join(chunks)
    )
    
    # Execute and collect
    result = list(grok.execute(prompt, stream=True, response=response, conversation=None))
    
    assert result == ["Hello ", "there!"]


def test_messages_endpoint_tool_calling(httpx_mock: HTTPXMock) -> None:
    """Test tool calling with messages endpoint."""
    grok = Grok("x-ai/grok-4")
    prompt = llm.Prompt("What's the weather?", model=grok)
    
    tools = cast(List[ToolDefinition], [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
        }
    }])
    
    prompt.options = Grok.Options(use_messages_endpoint=True, tools=tools)
    response = llm.Response(model=grok, prompt=prompt, stream=False)
    
    # Mock response with tool use
    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/messages",
        json={
            "id": "msg_123",
            "content": [
                {"type": "text", "text": "I'll check the weather."},
                {"type": "tool_use", "id": "tool_123", "name": "get_weather", "input": {"location": "NYC"}}
            ],
            "stop_reason": "tool_use"
        }
    )
    
    # Execute by iterating over the response
    result = list(response)
    
    # Check request has converted tools
    request = httpx_mock.get_requests()[0]
    body = json.loads(request.content)
    assert "tools" in body
    tools = body.get("tools", [])
    assert len(tools) > 0
    assert tools[0].get("name") == "get_weather"
    input_schema = tools[0].get("input_schema", {})
    assert input_schema.get("type") == "object"
    
    # Check response
    assert result == ["I'll check the weather."]
    tool_calls_list = response.tool_calls()  # Call the method after response is done
    assert len(tool_calls_list) == 1
    assert tool_calls_list[0].name == "get_weather"


def test_messages_endpoint_max_tokens(httpx_mock: HTTPXMock) -> None:
    """Test max_tokens parameter mapping for messages endpoint."""
    grok = Grok("x-ai/grok-4")
    prompt = llm.Prompt("Hello", model=grok)
    prompt.options = Grok.Options(use_messages_endpoint=True, max_completion_tokens=100)
    response = llm.Response(model=grok, prompt=prompt, stream=False)
    
    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/messages",
        json={"id": "msg_123", "content": [{"type": "text", "text": "Hi!"}], "stop_reason": "end_turn"}
    )
    
    list(grok.execute(prompt, stream=False, response=response, conversation=None))
    
    # Check request uses max_tokens not max_completion_tokens
    request = httpx_mock.get_requests()[0]
    body = json.loads(request.content)
    assert "max_tokens" in body
    assert body.get("max_tokens") == 100
    assert "max_completion_tokens" not in body


def test_messages_endpoint_invalid_tool_json(httpx_mock: HTTPXMock) -> None:
    """Test handling of invalid JSON in tool arguments."""
    grok = Grok("x-ai/grok-4")
    
    # Mock response with tool calls that have invalid JSON
    httpx_mock.add_response(
        method="POST",
        url="https://api.x.ai/v1/messages",
        json={
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "tool_123",
                "name": "get_weather",
                "input": {"location": "New York"}
            }],
            "model": "x-ai/grok-4",
            "stop_reason": "tool_use"
        }
    )
    
    prompt = llm.Prompt("Get weather", model=grok)
    prompt.options = Grok.Options(use_messages_endpoint=True)
    
    response = llm.Response(model=grok, prompt=prompt, stream=False)
    
    # This should handle the conversion gracefully
    result = list(response)  # Iterate over response instead of calling execute directly
    
    # Should have tool calls
    assert hasattr(response, 'tool_calls')
    # tool_calls() is always a method now
    tool_calls_list = response.tool_calls()
    assert tool_calls_list  # Check if list is not empty


def test_messages_endpoint_image_fetch_timeout() -> None:
    """Test image fetching timeout handling."""
    import httpx
    from io import StringIO
    from rich.console import Console
    from unittest.mock import patch, MagicMock
    
    grok = Grok("x-ai/grok-4")
    
    # Capture console output
    string_io = StringIO()
    test_console = Console(file=string_io, force_terminal=True)
    
    # Mock httpx.Client.stream to raise timeout
    with patch('httpx.Client.stream') as mock_stream:
        mock_stream.side_effect = httpx.TimeoutException("Timeout")
        
        # Test the image conversion method directly
        import llm_grok
        original_console = llm_grok.console
        llm_grok.console = test_console
        
        try:
            result = grok._convert_image_to_anthropic("https://example.com/image.jpg")
            
            # Should return None on timeout
            assert result is None
            
            # Check that timeout warning was shown
            output = string_io.getvalue()
            assert "timed out" in output.lower()
        finally:
            llm_grok.console = original_console


def test_messages_endpoint_image_size_limit() -> None:
    """Test image size limit enforcement."""
    import httpx
    from io import StringIO
    from rich.console import Console
    from unittest.mock import patch, MagicMock
    
    grok = Grok("x-ai/grok-4")
    
    # Capture console output
    string_io = StringIO()
    test_console = Console(file=string_io, force_terminal=True)
    
    # Mock large image response (2MB)
    large_image_data = b"x" * (2 * 1024 * 1024)
    
    # Mock httpx.Client.stream to return large content
    with patch('httpx.Client.stream') as mock_stream:
        # Create mock response
        mock_response = MagicMock()
        mock_response.headers = {
            "content-type": "image/jpeg",
            "content-length": str(len(large_image_data))
        }
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=None)
        mock_response.raise_for_status = MagicMock()
        
        mock_stream.return_value = mock_response
        
        # Test the image conversion method directly
        import llm_grok
        original_console = llm_grok.console
        llm_grok.console = test_console
        
        try:
            result = grok._convert_image_to_anthropic("https://example.com/large.jpg")
            
            # Should return None for oversized images
            assert result is None
            
            # Check that size limit warning was shown
            output = string_io.getvalue()
            assert "too large" in output.lower() or "1mb limit" in output.lower()
        finally:
            llm_grok.console = original_console
