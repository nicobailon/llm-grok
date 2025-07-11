"""Anthropic format handler for message conversion and SSE parsing."""

import json
from collections.abc import Iterator
from typing import Any, Optional, Union, cast

from llm_grok.constants import (
    FIRST_CHOICE_INDEX,
    SSE_DATA_PREFIX_LENGTH,
    SSE_DELIMITER,
    SSE_EVENT_PREFIX_LENGTH,
)
from llm_grok.types import (
    AnthropicRequest,
    AnthropicToolDefinition,
    Message,
    OpenAIResponse,
    OpenAIStreamChunk,
    ToolDefinition,
)

from .base import FormatHandler


class AnthropicFormatHandler(FormatHandler):
    """Handles Anthropic format operations and conversions."""

    def convert_messages_to_anthropic(self, openai_messages: list[Message]) -> AnthropicRequest:
        """Convert OpenAI messages to Anthropic format - not needed for Anthropic handler."""
        # This is handled by OpenAIFormatHandler
        raise NotImplementedError("Use OpenAIFormatHandler for OpenAI to Anthropic conversion")

    def convert_tools_to_anthropic(self, openai_tools: list[ToolDefinition]) -> list[AnthropicToolDefinition]:
        """Convert OpenAI tools to Anthropic format - not needed for Anthropic handler."""
        # This is handled by OpenAIFormatHandler
        raise NotImplementedError("Use OpenAIFormatHandler for tool conversion")

    def parse_openai_sse(self, buffer: str) -> tuple[Optional[dict[str, Any]], str]:
        """Parse OpenAI SSE - not needed for Anthropic handler."""
        raise NotImplementedError("Use OpenAIFormatHandler for OpenAI SSE parsing")

    def parse_anthropic_sse(self, buffer: str) -> tuple[Optional[tuple[str, dict[str, Any]]], str]:
        """Parse Anthropic SSE format and return (event_type, event_data) tuple and remaining buffer."""
        if "\n\n" not in buffer:
            return None, buffer

        message, remaining_buffer = buffer.split(SSE_DELIMITER, 1)
        lines = message.strip().split("\n")
        event_type = None
        event_data = None

        for line in lines:
            if line.startswith("event: "):
                event_type = line[SSE_EVENT_PREFIX_LENGTH:]
            elif line.startswith("data: "):
                data = line[SSE_DATA_PREFIX_LENGTH:]
                if data != "[DONE]":
                    try:
                        event_data = json.loads(data)
                    except json.JSONDecodeError:
                        continue

        if event_type and event_data:
            return (event_type, event_data), remaining_buffer
        return None, remaining_buffer

    def parse_sse_chunk(self, chunk: bytes) -> Iterator[Union[OpenAIStreamChunk, dict[str, Any]]]:
        """Parse SSE chunks - placeholder implementation."""
        # This would need a more complex implementation with buffer management
        yield {}

    def convert_anthropic_stream_chunk(self, event_type: str, event_data: dict[str, Any]) -> Optional[OpenAIStreamChunk]:
        """Convert Anthropic streaming event to OpenAI format chunk."""
        if event_type == "message_start":
            # Start of message - no content yet
            return cast(OpenAIStreamChunk, {
                "choices": [{
                    "index": FIRST_CHOICE_INDEX,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }]
            })

        elif event_type == "content_block_start":
            # Start of a content block
            block = event_data.get("content_block", {})
            if block.get("type") == "text":
                return cast(OpenAIStreamChunk, {
                    "choices": [{
                        "index": FIRST_CHOICE_INDEX,
                        "delta": {"content": ""},
                        "finish_reason": None
                    }]
                })
            elif block.get("type") == "tool_use":
                # Start of tool use
                return cast(OpenAIStreamChunk, {
                    "choices": [{
                        "index": FIRST_CHOICE_INDEX,
                        "delta": {
                            "tool_calls": [{
                                "index": event_data.get("index", 0),
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": ""
                                }
                            }]
                        },
                        "finish_reason": None
                    }]
                })

        elif event_type == "content_block_delta":
            # Content delta
            delta = event_data.get("delta", {})
            if delta.get("type") == "text_delta":
                return cast(OpenAIStreamChunk, {
                    "choices": [{
                        "index": FIRST_CHOICE_INDEX,
                        "delta": {"content": delta.get("text", "")},
                        "finish_reason": None
                    }]
                })
            elif delta.get("type") == "input_json_delta":
                # Tool input delta
                return cast(OpenAIStreamChunk, {
                    "choices": [{
                        "index": FIRST_CHOICE_INDEX,
                        "delta": {
                            "tool_calls": [{
                                "index": event_data.get("index", 0),
                                "function": {
                                    "arguments": delta.get("partial_json", "")
                                }
                            }]
                        },
                        "finish_reason": None
                    }]
                })

        elif event_type == "message_delta":
            # Message metadata updates (like stop_reason)
            delta = event_data.get("delta", {})
            if "stop_reason" in delta:
                return cast(OpenAIStreamChunk, {
                    "choices": [{
                        "index": FIRST_CHOICE_INDEX,
                        "delta": {},
                        "finish_reason": delta["stop_reason"]
                    }]
                })

        elif event_type == "message_stop":
            # Final message
            return cast(OpenAIStreamChunk, {
                "choices": [{
                    "index": FIRST_CHOICE_INDEX,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            })

        return None

    def convert_from_openai(self, openai_request: dict[str, Any]) -> dict[str, Any]:
        """Convert a full OpenAI request to Anthropic format.
        
        Args:
            openai_request: Complete OpenAI request including messages, tools, etc.
            
        Returns:
            Complete Anthropic request ready for the API
        """
        # Extract messages and convert them
        openai_messages = openai_request.get("messages", [])

        # Separate system messages from user/assistant messages
        system_messages = []
        conversation_messages = []

        for msg in openai_messages:
            if msg["role"] == "system":
                system_messages.append(msg["content"])
            else:
                conversation_messages.append(msg)

        # Build Anthropic request
        anthropic_request: dict[str, Any] = {
            "messages": []
        }

        # Add system message if present
        if system_messages:
            anthropic_request["system"] = "\n\n".join(system_messages)

        # Convert conversation messages
        for msg in conversation_messages:
            anthropic_msg: dict[str, Any] = {
                "role": msg["role"]
            }

            # Handle content conversion
            if isinstance(msg.get("content"), str):
                anthropic_msg["content"] = msg["content"]
            elif isinstance(msg.get("content"), list):
                # Multimodal content - convert format
                anthropic_content = []
                for part in msg["content"]:
                    if part["type"] == "text":
                        anthropic_content.append({
                            "type": "text",
                            "text": part["text"]
                        })
                    elif part["type"] == "image_url":
                        anthropic_content.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": part["image_url"]["url"]
                            }
                        })
                anthropic_msg["content"] = anthropic_content

            # Handle tool messages
            if msg.get("role") == "tool":
                anthropic_msg["role"] = "user"
                anthropic_msg["content"] = [{
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", "")
                }]

            # Handle assistant messages with tool calls
            if "tool_calls" in msg:
                content_parts = []
                if msg.get("content"):
                    content_parts.append({
                        "type": "text",
                        "text": msg["content"]
                    })

                for tool_call in msg["tool_calls"]:
                    func = tool_call["function"]
                    content_parts.append({
                        "type": "tool_use",
                        "id": tool_call["id"],
                        "name": func["name"],
                        "input": json.loads(func.get("arguments", "{}")) if isinstance(func.get("arguments"), str) else func.get("arguments", {})
                    })

                anthropic_msg["content"] = content_parts

            anthropic_request["messages"].append(anthropic_msg)

        # Convert tools if present
        if "tools" in openai_request:
            anthropic_request["tools"] = []
            for tool in openai_request["tools"]:
                if tool["type"] == "function":
                    func = tool["function"]
                    anthropic_request["tools"].append({
                        "type": "function",
                        "function": {
                            "name": func["name"],
                            "description": func.get("description", ""),
                            "input_schema": func.get("parameters", {})
                        }
                    })

        # Copy other parameters
        for key in ["temperature", "max_tokens", "tool_choice"]:
            if key in openai_request:
                anthropic_request[key] = openai_request[key]

        return anthropic_request

    def convert_from_anthropic_response(self, anthropic_response: dict[str, Any]) -> OpenAIResponse:
        """Convert Anthropic response to OpenAI format."""
        # Extract content from Anthropic response
        content_parts = anthropic_response.get("content", [])
        text_content = ""
        tool_calls = []

        for i, part in enumerate(content_parts):
            if part.get("type") == "text":
                text_content += part.get("text", "")
            elif part.get("type") == "tool_use":
                # Convert Anthropic tool use to OpenAI tool call
                tool_call = {
                    "id": part.get("id", f"call_{i}"),
                    "type": "function",
                    "function": {
                        "name": part.get("name", ""),
                        "arguments": part.get("input", {})
                    }
                }
                if "arguments" in tool_call["function"] and isinstance(tool_call["function"]["arguments"], dict):
                    tool_call["function"]["arguments"] = json.dumps(tool_call["function"]["arguments"])
                tool_calls.append(tool_call)

        # Build OpenAI-style response
        message: dict[str, Any] = {
            "role": "assistant",
            "content": text_content if text_content else None
        }

        if tool_calls:
            message["tool_calls"] = tool_calls

        return cast(OpenAIResponse, {
            "id": anthropic_response.get("id", ""),
            "object": "chat.completion",
            "created": int(anthropic_response.get("created_at", 0)),
            "model": anthropic_response.get("model", self.model_id),
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": anthropic_response.get("stop_reason", "stop")
            }],
            "usage": anthropic_response.get("usage", {})
        })
