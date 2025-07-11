"""Unit tests for format handlers."""

import json
from typing import Any, Dict, List, cast

import pytest

from llm_grok.formats import OpenAIFormatHandler, AnthropicFormatHandler
from llm_grok.types import Message, ToolDefinition


class TestOpenAIFormatHandler:
    """Test OpenAI format handler functionality."""
    
    def test_convert_simple_messages_to_anthropic(self) -> None:
        """Test converting simple OpenAI messages to Anthropic format."""
        handler = OpenAIFormatHandler("grok-4")
        
        messages: List[Message] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        
        result = handler.convert_messages_to_anthropic(messages)
        
        assert result.get("system") == "You are helpful."
        messages_list = result.get("messages", [])
        assert len(messages_list) == 2
        assert messages_list[0]["role"] == "user"
        # Content is converted to Anthropic format (list of blocks)
        assert isinstance(messages_list[0]["content"], list)
        assert len(messages_list[0]["content"]) == 1
        content_block_0 = cast(Dict[str, Any], messages_list[0]["content"][0])
        assert content_block_0["type"] == "text"
        assert content_block_0["text"] == "Hello!"
        assert messages_list[1]["role"] == "assistant"
        assert isinstance(messages_list[1]["content"], list)
        # Access the text content properly
        content_block_1 = cast(Dict[str, Any], messages_list[1]["content"][0])
        assert content_block_1["type"] == "text"
        assert content_block_1["text"] == "Hi there!"
    
    def test_convert_multimodal_messages_to_anthropic(self) -> None:
        """Test converting multimodal OpenAI messages to Anthropic format."""
        handler = OpenAIFormatHandler("grok-4")
        
        messages: List[Message] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
                ]
            }
        ]
        
        result = handler.convert_messages_to_anthropic(messages)
        
        messages_list = result.get("messages", [])
        assert len(messages_list) == 1
        content = messages_list[0]["content"]
        assert isinstance(content, list)
        # Image URLs are not supported by Anthropic, so only text is converted
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "What's in this image?"
    
    def test_convert_tools_to_anthropic(self) -> None:
        """Test converting OpenAI tools to Anthropic format."""
        handler = OpenAIFormatHandler("grok-4")
        
        tools: List[ToolDefinition] = [{
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
        }]
        
        result = handler.convert_tools_to_anthropic(tools)
        
        assert len(result) == 1
        assert result[0]["name"] == "get_weather"
        assert result[0]["description"] == "Get weather for a location"
        assert result[0]["input_schema"] == tools[0]["function"]["parameters"]
    
    def test_convert_from_anthropic_response(self) -> None:
        """Test converting Anthropic response to OpenAI format."""
        handler = OpenAIFormatHandler("grok-4")
        
        anthropic_response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "grok-4",
            "stop_reason": "end_turn"
        }
        
        result = handler.convert_from_anthropic_response(anthropic_response)
        
        assert result.get("id") == "msg_123"
        assert result.get("model") == "grok-4"
        choices = result.get("choices", [])
        assert len(choices) > 0
        first_choice = choices[0]
        assert first_choice.get("message", {}).get("role") == "assistant"
        assert first_choice.get("message", {}).get("content") == "Hello!"
        # Anthropic's end_turn is preserved as is
        assert first_choice.get("finish_reason") == "end_turn"
    
    def test_parse_openai_sse(self) -> None:
        """Test parsing OpenAI SSE format."""
        handler = OpenAIFormatHandler("grok-4")
        
        # Valid SSE event - parse_sse returns tuple (parsed_data, remaining_buffer)
        event, remaining = handler.parse_sse('data: {"id":"123","choices":[{"delta":{"content":"Hi"}}]}\n\n')
        assert event is not None
        assert event.get("id") == "123"
        
        # [DONE] event
        event, remaining = handler.parse_sse("data: [DONE]\n\n")
        assert event is not None
        assert event.get("done") is True
        
        # Invalid JSON
        event, remaining = handler.parse_sse("data: invalid json\n\n")
        assert event is None


class TestAnthropicFormatHandler:
    """Test Anthropic format handler functionality."""
    
    def test_parse_anthropic_sse(self) -> None:
        """Test parsing Anthropic SSE format."""
        handler = AnthropicFormatHandler("grok-4")
        
        # Valid event - parse_anthropic_sse returns a tuple
        result = handler.parse_anthropic_sse('event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"Hi"}}\n\n')
        event_tuple, remaining = result
        assert event_tuple is not None
        event_type, event_data = event_tuple
        assert event_type == "content_block_delta"
        assert event_data.get("type") == "content_block_delta"
        
        # Invalid JSON
        result = handler.parse_anthropic_sse("event: test\ndata: invalid\n\n")
        event_tuple, remaining = result
        assert event_tuple is None
    
    def test_convert_from_anthropic_response(self) -> None:
        """Test converting Anthropic response (pass-through)."""
        handler = AnthropicFormatHandler("grok-4")
        
        response = {
            "id": "msg_123",
            "content": [{"type": "text", "text": "Hello"}]
        }
        
        # Should convert to OpenAI format
        result = handler.convert_from_anthropic_response(response)
        assert result.get("id") == "msg_123"
        assert result.get("object") == "chat.completion"
        assert "choices" in result
        assert len(result["choices"]) == 1
        message = result["choices"][0]["message"]
        assert message.get("role") == "assistant"
        assert message.get("content") == "Hello"