"""Consolidated mock data and utilities for testing."""
import base64
from typing import Any, Dict, List, Optional, Union

# Common test data constants
TEST_API_KEY = "test-api-key-123"
TEST_MODEL_ID = "x-ai/grok-4"

# API URLs
CHAT_COMPLETIONS_URL = "https://api.x.ai/v1/chat/completions"
MESSAGES_URL = "https://api.x.ai/v1/messages"

# Sample image data
SAMPLE_JPEG_BASE64 = "/9j/4AAQSkZJRgABAQEAAAAAAAD/2wBDAAEBAQ=="
SAMPLE_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
SAMPLE_IMAGE_URL = "https://example.com/image.jpg"
SAMPLE_DATA_URL = f"data:image/png;base64,{SAMPLE_PNG_BASE64}"
INVALID_BASE64 = "not-valid-base64!"

# Common error responses
ERROR_RESPONSES = {
    "invalid_request": {
        "error": {
            "message": "Invalid request",
            "type": "invalid_request_error",
            "code": "invalid_api_key"
        }
    },
    "rate_limit": {
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
            "code": "rate_limit_exceeded"
        }
    },
    "authentication_error": {
        "error": {
            "message": "Invalid API key",
            "type": "authentication_error",
            "code": "invalid_api_key"
        }
    }
}

# Sample tool definitions
SAMPLE_TOOLS = {
    "get_weather": {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
}

# Sample tool calls
SAMPLE_TOOL_CALLS = [{
    "id": "call_123",
    "type": "function",
    "function": {
        "name": "get_weather",
        "arguments": '{"location": "New York"}'
    }
}]


def create_streaming_chunks(content: Union[str, List[str]]) -> bytes:
    """Create SSE streaming chunks for OpenAI format."""
    if isinstance(content, str):
        content = [content]
    
    chunks = []
    for text in content:
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "model": TEST_MODEL_ID,
            "choices": [{
                "delta": {"content": text},
                "index": 0,
                "finish_reason": None
            }]
        }
        chunks.append(f"data: {json.dumps(chunk)}\n\n".encode())
    
    chunks.append(b"data: [DONE]\n\n")
    return b"".join(chunks)


def create_anthropic_streaming_chunks(content: List[str]) -> str:
    """Create SSE streaming chunks for Anthropic format."""
    chunks = []
    
    # Start event
    chunks.append('data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","model":"' + TEST_MODEL_ID + '"}}\n\n')
    
    # Content chunks
    for text in content:
        chunk = {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text}
        }
        chunks.append(f"data: {json.dumps(chunk)}\n\n")
    
    # End event
    chunks.append('data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\n\n')
    
    return "".join(chunks)


def create_chat_completion_response(
    content: str,
    tool_calls: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Create a mock chat completion response."""
    response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": TEST_MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }]
    }
    
    if tool_calls:
        response["choices"][0]["message"]["tool_calls"] = tool_calls
        response["choices"][0]["finish_reason"] = "tool_calls"
    
    return response


def create_messages_response(content: str) -> Dict[str, Any]:
    """Create a mock Anthropic messages response."""
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": TEST_MODEL_ID,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20}
    }


def create_multimodal_message(text: str, image_url: str) -> Dict[str, Any]:
    """Create a multimodal message with text and image."""
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }


def create_test_request(
    prompt: str,
    model: str = TEST_MODEL_ID,
    stream: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Create a test request body."""
    body: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream
    }
    
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    
    return body


# Import json for streaming functions
import json