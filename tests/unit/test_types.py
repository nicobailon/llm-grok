"""Tests for type definitions in llm_grok.types module."""

from typing import get_type_hints, Literal, Dict

from llm_grok.types import (
    ImageContent,
    TextContent,
    FunctionCallDetails,
    ToolCall,
    ToolCallWithIndex,
    BaseMessage,
    Message,
    ToolDefinition,
    AnthropicImage,
    AnthropicMessage,
    AnthropicRequest,
    ModelInfo,
)


class TestOpenAITypes:
    """Test suite for OpenAI API type definitions."""
    
    def test_image_content_structure(self) -> None:
        """Test ImageContent TypedDict structure."""
        # Create valid image content
        image_content: ImageContent = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.jpg"}
        }
        
        # Verify structure
        assert image_content["type"] == "image_url"
        assert "url" in image_content["image_url"]
        
        # Get type hints to verify typing
        hints = get_type_hints(ImageContent)
        assert hints["type"] == Literal["image_url"]
        assert hints["image_url"] == dict[str, str]
    
    def test_text_content_structure(self) -> None:
        """Test TextContent TypedDict structure."""
        text_content: TextContent = {
            "type": "text",
            "text": "Hello, world!"
        }
        
        assert text_content["type"] == "text"
        assert isinstance(text_content["text"], str)
        
        hints = get_type_hints(TextContent)
        assert hints["type"] == Literal["text"]
        assert hints["text"] == str
    
    def test_function_call_details(self) -> None:
        """Test FunctionCallDetails structure."""
        function_details: FunctionCallDetails = {
            "name": "get_weather",
            "arguments": '{"location": "San Francisco"}'
        }
        
        assert isinstance(function_details["name"], str)
        assert isinstance(function_details["arguments"], str)
        
        # Verify arguments is meant to be JSON string
        import json
        parsed_args = json.loads(function_details["arguments"])
        assert parsed_args["location"] == "San Francisco"
    
    def test_tool_call_structure(self) -> None:
        """Test ToolCall structure with all fields."""
        # Basic tool call without index
        tool_call: ToolCall = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "calculate",
                "arguments": '{"x": 5, "y": 3}'
            }
        }
        
        assert tool_call["id"] == "call_123"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "calculate"
        
        # Tool call with index for streaming accumulation
        tool_call_with_index: ToolCallWithIndex = {
            "id": "call_456",
            "type": "function",
            "function": {"name": "test", "arguments": "{}"},
            "index": 0
        }
        assert tool_call_with_index["index"] == 0
    
    def test_message_variants(self) -> None:
        """Test different Message format variants."""
        # Simple text message (using BaseMessage for required fields only)
        simple_message: BaseMessage = {
            "role": "user",
            "content": "Hello"
        }
        assert simple_message["role"] == "user"
        assert simple_message["content"] == "Hello"
        
        # Multimodal message
        multimodal_message: BaseMessage = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's this?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }
        assert isinstance(multimodal_message["content"], list)
        assert len(multimodal_message["content"]) == 2
        
        # Assistant message with tool calls
        assistant_message: Message = {
            "role": "assistant",
            "content": "I'll help you with that.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"}
                }
            ]
        }
        assert "tool_calls" in assistant_message
        assert len(assistant_message["tool_calls"]) == 1
    
    def test_tool_definition(self) -> None:
        """Test ToolDefinition structure."""
        tool_def: ToolDefinition = {
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
        
        assert tool_def["type"] == "function"
        assert "name" in tool_def["function"]
        assert "parameters" in tool_def["function"]


class TestAnthropicTypes:
    """Test suite for Anthropic API type definitions."""
    
    def test_anthropic_image_structure(self) -> None:
        """Test AnthropicImage with source."""
        anthropic_image: AnthropicImage = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "base64encodeddata..."
            }
        }
        
        assert anthropic_image["type"] == "image"
        assert anthropic_image["source"]["type"] == "base64"
        assert anthropic_image["source"]["media_type"] == "image/jpeg"
    
    def test_anthropic_message_structure(self) -> None:
        """Test AnthropicMessage with mixed content."""
        message: AnthropicMessage = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "..."
                    }
                }
            ]
        }
        
        assert message["role"] == "user"
        assert len(message["content"]) == 2
        assert message["content"][0]["type"] == "text"
        assert message["content"][1]["type"] == "image"
    
    def test_anthropic_request_structure(self) -> None:
        """Test AnthropicRequest with optional fields."""
        # Minimal request
        minimal_request: AnthropicRequest = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}]
                }
            ],
            "model": "grok-4"
        }
        
        assert len(minimal_request["messages"]) == 1
        assert minimal_request["model"] == "grok-4"
        
        # Full request
        full_request: AnthropicRequest = {
            "messages": [
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": "Hi"}]
                }
            ],
            "system": "You are a helpful assistant",
            "model": "grok-4",
            "max_tokens": 1000,
            "temperature": 0.7,
            "tools": [
                {
                    "name": "calculator",
                    "description": "Performs calculations",
                    "input_schema": {"type": "object", "properties": {}}
                }
            ],
            "tool_choice": "auto"
        }
        
        assert full_request["system"] == "You are a helpful assistant"
        assert full_request["temperature"] == 0.7
        assert full_request["tool_choice"] == "auto"


class TestModelInfoType:
    """Test suite for ModelInfo type."""
    
    def test_model_info_structure(self) -> None:
        """Test ModelInfo TypedDict with all fields."""
        model_info: ModelInfo = {
            "context_window": 256000,
            "supports_vision": True,
            "supports_tools": True,
            "max_output_tokens": 8192,
            "pricing_tier": "standard"
        }
        
        assert model_info["context_window"] == 256000
        assert model_info["supports_vision"] is True
        assert model_info["supports_tools"] is True
        assert model_info["max_output_tokens"] == 8192
        assert model_info["pricing_tier"] == "standard"
    
    def test_model_info_partial(self) -> None:
        """Test ModelInfo with all required fields."""
        # Now all fields are required in ModelInfo
        partial_info: ModelInfo = {
            "context_window": 128000,
            "supports_vision": False,
            "supports_tools": False,
            "max_output_tokens": 4096,
            "pricing_tier": "standard"
        }
        
        assert partial_info["context_window"] == 128000
        assert partial_info["supports_vision"] is False
        assert partial_info["supports_tools"] is False
        assert partial_info["max_output_tokens"] == 4096
        assert partial_info["pricing_tier"] == "standard"


def test_type_annotations_complete() -> None:
    """Verify all exported types have proper annotations."""
    from llm_grok import types
    
    # Check that all exported types are properly typed
    for name in types.__all__:
        obj = getattr(types, name)
        if isinstance(obj, type) and issubclass(obj, dict):
            # For TypedDict classes, verify they have __annotations__
            assert hasattr(obj, "__annotations__"), f"{name} missing type annotations"
            assert len(obj.__annotations__) > 0, f"{name} has empty annotations"