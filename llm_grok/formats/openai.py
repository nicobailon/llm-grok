"""OpenAI format handler for message conversion and SSE parsing."""

import json
import warnings
from collections.abc import Iterator
from typing import Any, Literal, Optional, Union, cast

from llm_grok.constants import (
    FIRST_CHOICE_INDEX,
    SSE_DATA_PREFIX_LENGTH,
    SSE_DELIMITER,
    SSE_EVENT_PREFIX_LENGTH,
)
from llm_grok.exceptions import ValidationError
from llm_grok.types import (
    AnthropicImage,
    AnthropicImageSource,
    AnthropicMessage,
    AnthropicRequest,
    AnthropicTextBlock,
    AnthropicToolChoice,
    AnthropicToolDefinition,
    AnthropicToolUse,
    ImageContent,
    Message,
    OpenAIResponse,
    OpenAIStreamChunk,
    TextContent,
    ToolCall,
    ToolChoice,
    ToolDefinition,
)

from .base import FormatHandler


class OpenAIFormatHandler(FormatHandler):
    """Handles OpenAI format operations and conversions."""

    def parse_sse(self, buffer: str) -> tuple[Optional[dict[str, Any]], str]:
        """Parse OpenAI SSE format and return parsed data and remaining buffer."""
        if "\n\n" not in buffer:
            return None, buffer

        message, remaining_buffer = buffer.split(SSE_DELIMITER, 1)
        if message.startswith("data: "):
            data = message[6:]
            if data == "[DONE]":
                return {"done": True}, remaining_buffer
            try:
                return json.loads(data), remaining_buffer
            except json.JSONDecodeError:
                pass
        return None, remaining_buffer

    def parse_openai_sse(self, buffer: str) -> tuple[Optional[dict[str, Any]], str]:
        """Alias for parse_sse for compatibility.
        
        .. deprecated:: 3.0
            Use parse_sse() directly instead.
        """
        warnings.warn(
            "parse_openai_sse is deprecated and will be removed in v4.0. "
            "Use parse_sse() directly.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.parse_sse(buffer)

    def convert_messages_to_anthropic(self, openai_messages: list[Message]) -> AnthropicRequest:
        """Convert OpenAI-style messages to Anthropic format.
        
        Args:
            openai_messages: List of OpenAI-format messages
            
        Returns:
            Dict with 'messages', 'system' (optional), suitable for Anthropic API
        """
        system_prompts = self._extract_system_prompts(openai_messages)
        anthropic_messages = self._convert_user_assistant_messages(openai_messages)

        result: AnthropicRequest = {"messages": anthropic_messages}
        if system_prompts:
            result["system"] = "\n\n".join(system_prompts)

        return result

    def _extract_system_prompts(self, messages: list[Message]) -> list[str]:
        """Extract and combine system messages from OpenAI messages.
        
        Args:
            messages: List of OpenAI-format messages
            
        Returns:
            List of system prompt strings
        """
        system_prompts = []

        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    system_prompts.append(content)
                else:
                    # If content is multimodal, extract text parts
                    for part in content:
                        if part["type"] == "text":
                            system_prompts.append(part["text"])

        return system_prompts

    def _convert_user_assistant_messages(self, messages: list[Message]) -> list[AnthropicMessage]:
        """Convert user and assistant messages to Anthropic format.
        
        Args:
            messages: List of OpenAI-format messages
            
        Returns:
            List of Anthropic-format messages
        """
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role", "")
            if role in ["user", "assistant"]:
                anthropic_msg = self._convert_single_message(msg)
                anthropic_messages.append(anthropic_msg)

        return anthropic_messages

    def _convert_single_message(self, msg: Message) -> AnthropicMessage:
        """Convert a single OpenAI message to Anthropic format.
        
        Args:
            msg: OpenAI-format message
            
        Returns:
            Anthropic-format message
        """
        role = msg.get("role", "")
        content = msg.get("content", "")
        anthropic_role = cast(Literal["user", "assistant"], role)
        anthropic_msg: AnthropicMessage = {"role": anthropic_role, "content": []}

        if isinstance(content, str):
            # Simple text content
            text_block: AnthropicTextBlock = {
                "type": "text",
                "text": content
            }
            anthropic_msg["content"] = [text_block]
        else:
            # Multimodal content
            converted_content = self._convert_multimodal_content(content)
            # Need to cast because we're initially creating with empty list
            anthropic_msg["content"] = cast(list[Union[AnthropicTextBlock, AnthropicImage, AnthropicToolUse]], converted_content)

        # Handle tool calls in assistant messages
        if role == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
            tool_uses = self._convert_tool_calls(msg["tool_calls"])
            # Extend the content list with tool uses
            anthropic_msg["content"].extend(tool_uses)

        return anthropic_msg

    def _convert_multimodal_content(self, content: list[Union[TextContent, ImageContent]]) -> list[Union[AnthropicTextBlock, AnthropicImage]]:
        """Convert multimodal content to Anthropic format.
        
        Args:
            content: List of OpenAI content items
            
        Returns:
            List of Anthropic content blocks
        """
        anthropic_content: list[Union[AnthropicTextBlock, AnthropicImage]] = []

        for part in content:
            if part["type"] == "text":
                text_block: AnthropicTextBlock = {
                    "type": "text",
                    "text": part["text"]
                }
                anthropic_content.append(text_block)
            elif part["type"] == "image_url":
                image_block = self._convert_image_content(part)
                if image_block:
                    anthropic_content.append(image_block)

        return anthropic_content

    def _convert_image_content(self, image_content: ImageContent) -> Optional[AnthropicImage]:
        """Convert OpenAI image content to Anthropic format.
        
        Args:
            image_content: OpenAI image content
            
        Returns:
            Anthropic image block or None if URL is not supported
        """
        image_url = image_content["image_url"]["url"]

        # Validate URL for security if it's not a data URL
        if not image_url.startswith("data:"):
            try:
                # Validate the URL to prevent SSRF attacks
                validated_url = self.validate_image_url(image_url)
                # Anthropic doesn't support direct URLs, so skip after validation
                return None
            except ValidationError:
                # Log and skip invalid URLs
                return None

        # Extract base64 data from data URL
        parts = image_url.split(",", 1)
        if len(parts) != 2:
            return None

        media_type_part = parts[0].split(";")[0].split(":")[1]
        base64_data = parts[1]

        image_source: AnthropicImageSource = {
            "type": "base64",
            "media_type": media_type_part,
            "data": base64_data
        }
        image_block: AnthropicImage = {
            "type": "image",
            "source": image_source
        }

        return image_block

    def _convert_tool_calls(self, tool_calls: list[ToolCall]) -> list[AnthropicToolUse]:
        """Convert OpenAI tool calls to Anthropic tool use format.
        
        Args:
            tool_calls: List of OpenAI tool calls
            
        Returns:
            List of Anthropic tool use blocks
        """
        tool_uses: list[AnthropicToolUse] = []

        for tool_call in tool_calls:
            if "function" in tool_call and tool_call["function"]:
                try:
                    arguments = json.loads(tool_call["function"].get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {}

                tool_use: AnthropicToolUse = {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "input": arguments
                }
                tool_uses.append(tool_use)

        return tool_uses

    def convert_tools_to_anthropic(self, openai_tools: list[ToolDefinition]) -> list[AnthropicToolDefinition]:
        """Convert OpenAI tool definitions to Anthropic format.
        
        Args:
            openai_tools: List of OpenAI-format tool definitions
            
        Returns:
            List of Anthropic-format tool definitions
        """
        anthropic_tools = []

        for tool in openai_tools:
            if tool["type"] == "function":
                func = tool["function"]
                anthropic_tool: AnthropicToolDefinition = {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                }
                anthropic_tools.append(anthropic_tool)

        return anthropic_tools

    def convert_tool_choice_to_anthropic(self, tool_choice: Union[Literal["auto", "none"], ToolChoice]) -> Optional[Union[Literal["auto"], AnthropicToolChoice]]:
        """Convert OpenAI tool choice to Anthropic format.
        
        Args:
            tool_choice: OpenAI tool choice specification
            
        Returns:
            Anthropic tool choice or None
        """
        if tool_choice == "auto":
            return "auto"
        elif tool_choice == "none":
            return None  # Anthropic doesn't send tool_choice for none
        elif isinstance(tool_choice, dict) and "function" in tool_choice:
            # Convert specific function choice
            return {
                "type": "tool",
                "name": tool_choice["function"]["name"]
            }
        return None

    def convert_from_anthropic_response(self, anthropic_response: dict[str, Any]) -> OpenAIResponse:
        """Convert Anthropic response to OpenAI format."""
        import time

        openai_response = {
            "id": anthropic_response.get("id", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": anthropic_response.get("model", self.model_id),
            "choices": [],
            "usage": self._convert_usage(anthropic_response.get("usage", {}))
        }

        # Convert content
        content = anthropic_response.get("content", [])
        if isinstance(content, list):
            # Extract text content
            text_parts = [part["text"] for part in content if part["type"] == "text"]
            message_content = "".join(text_parts)

            # Extract tool uses
            tool_uses = [part for part in content if part["type"] == "tool_use"]
            tool_calls = None
            if tool_uses:
                tool_calls = []
                for i, tool_use in enumerate(tool_uses):
                    tool_calls.append({
                        "id": tool_use.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tool_use["name"],
                            "arguments": json.dumps(tool_use.get("input", {}))
                        }
                    })
        else:
            message_content = content
            tool_calls = None

        message: dict[str, Any] = {
            "role": "assistant",
            "content": message_content
        }
        if tool_calls:
            message["tool_calls"] = tool_calls

        openai_response["choices"].append({
            "index": FIRST_CHOICE_INDEX,
            "message": message,
            "finish_reason": anthropic_response.get("stop_reason", "stop")
        })

        return cast(OpenAIResponse, openai_response)

    def parse_anthropic_sse(self, buffer: str) -> tuple[Optional[tuple[str, dict[str, Any]]], str]:
        """Parse Anthropic SSE format and return event data and remaining buffer."""
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

    def convert_anthropic_stream_chunk(self, event_type: str, event_data: dict[str, Any]) -> Optional[OpenAIStreamChunk]:
        """Convert Anthropic streaming event to OpenAI format chunk."""
        import time

        if event_type == "message_start":
            # Initial message metadata
            return cast(OpenAIStreamChunk, {
                "id": event_data.get("message", {}).get("id", ""),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": event_data.get("message", {}).get("model", self.model_id),
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

    def parse_sse_chunk(self, chunk: bytes) -> Iterator[Union[OpenAIStreamChunk, dict[str, Any]]]:
        """Parse Server-Sent Events chunks from either API format.
        
        This implementation focuses on OpenAI format parsing.
        """
        text = chunk.decode("utf-8", errors="ignore")
        buffer = text

        while buffer:
            parsed, buffer = self.parse_openai_sse(buffer)
            if parsed:
                yield parsed
            else:
                break

    def _convert_usage(self, anthropic_usage: dict[str, Any]) -> dict[str, int]:
        """Convert Anthropic usage to OpenAI format."""
        input_tokens = anthropic_usage.get("input_tokens", 0)
        output_tokens = anthropic_usage.get("output_tokens", 0)
        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }

    def convert_from_anthropic(self, anthropic_data: dict[str, Any]) -> dict[str, Any]:
        """Convert Anthropic format data to OpenAI format.
        
        Args:
            anthropic_data: Anthropic-format request data
            
        Returns:
            OpenAI-format request data
        """
        result: dict[str, Any] = {"messages": []}

        # Convert system message if present
        if "system" in anthropic_data and anthropic_data["system"]:
            result["messages"].append({
                "role": "system",
                "content": anthropic_data["system"]
            })

        # Convert messages
        for msg in anthropic_data.get("messages", []):
            openai_msg = self._convert_anthropic_message_to_openai(msg)
            if "role" in msg and msg["role"] == "user":
                # Handle tool results in user messages
                if isinstance(msg.get("content"), list):
                    tool_results = []
                    regular_content = []
                    for part in msg["content"]:
                        if part.get("type") == "tool_result":
                            tool_results.append(part)
                        else:
                            regular_content.append(part)

                    # Add regular user message if there's non-tool content
                    if regular_content:
                        openai_msg["content"] = self._convert_anthropic_content_to_openai(regular_content)
                        result["messages"].append(openai_msg)

                    # Add tool messages separately
                    for tool_result in tool_results:
                        result["messages"].append({
                            "role": "tool",
                            "tool_call_id": tool_result.get("tool_use_id", ""),
                            "content": json.dumps(tool_result.get("content", {}))
                            if isinstance(tool_result.get("content"), dict)
                            else str(tool_result.get("content", ""))
                        })
                else:
                    result["messages"].append(openai_msg)
            else:
                result["messages"].append(openai_msg)

        # Convert tools if present
        if "tools" in anthropic_data:
            result["tools"] = self._convert_anthropic_tools_to_openai(anthropic_data["tools"])

        # Copy other parameters
        for key in ["temperature", "max_tokens", "stop_sequences"]:
            if key in anthropic_data:
                openai_key = "max_completion_tokens" if key == "max_tokens" else key
                openai_key = "stop" if key == "stop_sequences" else openai_key
                result[openai_key] = anthropic_data[key]

        return result

    def _convert_anthropic_message_to_openai(self, msg: AnthropicMessage) -> Message:
        """Convert a single Anthropic message to OpenAI format."""
        openai_msg: Message
        
        if isinstance(msg["content"], str):
            openai_msg = {"role": msg["role"], "content": msg["content"]}
        elif isinstance(msg["content"], list):
            # Handle multimodal content
            content = self._convert_anthropic_content_to_openai(msg["content"])

            # Extract tool calls if present
            tool_uses = [cast(AnthropicToolUse, part) for part in msg["content"] if part.get("type") == "tool_use"]
            if tool_uses and msg["role"] == "assistant":
                # Set content to text parts only
                text_parts = [part for part in msg["content"] if part.get("type") != "tool_use"]
                if text_parts:
                    content_value = self._convert_anthropic_content_to_openai(text_parts)
                else:
                    content_value = ""

                # Create message with tool calls
                openai_msg = {
                    "role": msg["role"],
                    "content": content_value,
                    "tool_calls": [
                        {
                            "id": tool_use["id"],
                            "type": "function",
                            "function": {
                                "name": tool_use["name"],
                                "arguments": json.dumps(tool_use.get("input", {}))
                            }
                        }
                        for tool_use in tool_uses
                    ]
                }
            else:
                openai_msg = {"role": msg["role"], "content": content}
        else:
            # Fallback for unexpected content types
            openai_msg = {"role": msg["role"], "content": ""}

        return openai_msg

    def _convert_anthropic_content_to_openai(
        self, content: list[Union[AnthropicTextBlock, AnthropicImage, AnthropicToolUse]]
    ) -> Union[str, list[Union[TextContent, ImageContent]]]:
        """Convert Anthropic content blocks to OpenAI format."""
        # Filter out tool_use blocks which are handled separately
        filtered_content = [part for part in content if part.get("type") != "tool_use"]

        # If all content is text, return as string
        if all(part.get("type") == "text" for part in filtered_content):
            text_parts = [part for part in filtered_content if part.get("type") == "text"]
            return "".join(cast(AnthropicTextBlock, part)["text"] for part in text_parts)

        # Otherwise, return multimodal array
        openai_content: list[Union[TextContent, ImageContent]] = []
        for part in filtered_content:
            if part.get("type") == "text":
                text_part = cast(AnthropicTextBlock, part)
                text_content: TextContent = {
                    "type": "text",
                    "text": text_part["text"]
                }
                openai_content.append(text_content)
            elif part.get("type") == "image":
                image_part = cast(AnthropicImage, part)
                source = image_part["source"]
                if source["type"] == "base64":
                    # Convert to data URL
                    media_type = source["media_type"]
                    data = source["data"]
                    image_content: ImageContent = {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{data}"
                        }
                    }
                    openai_content.append(image_content)
                elif source["type"] == "url":
                    image_content = {
                        "type": "image_url",
                        "image_url": {
                            "url": source["url"]
                        }
                    }
                    openai_content.append(image_content)

        return openai_content

    def _convert_anthropic_tools_to_openai(self, anthropic_tools: list[AnthropicToolDefinition]) -> list[ToolDefinition]:
        """Convert Anthropic tool definitions to OpenAI format."""
        openai_tools: list[ToolDefinition] = []

        for tool in anthropic_tools:
            # Support both direct tool format and wrapped format
            if "function" in tool:
                # Already in OpenAI format
                openai_tools.append(tool)  # type: ignore
            else:
                # Convert from Anthropic format
                openai_tool: ToolDefinition = {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {"type": "object", "properties": {}})
                    }
                }
                openai_tools.append(openai_tool)

        return openai_tools
