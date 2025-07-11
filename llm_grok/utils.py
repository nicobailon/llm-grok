"""Simple utility functions for format conversion and SSE parsing."""

import json
from collections.abc import Iterator
from typing import Any, Dict, List, Optional, Tuple


def parse_openai_sse(chunk: bytes) -> Iterator[Dict[str, Any]]:
    """Parse OpenAI Server-Sent Events chunks."""
    text = chunk.decode('utf-8', errors='ignore')

    for line in text.split('\n'):
        if line.startswith('data: '):
            data = line[6:].strip()
            if data == '[DONE]':
                continue
            try:
                yield json.loads(data)
            except json.JSONDecodeError:
                continue


def parse_anthropic_sse(chunk: bytes) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Parse Anthropic Server-Sent Events chunks."""
    text = chunk.decode('utf-8', errors='ignore')

    event_type = None
    for line in text.split('\n'):
        if line.startswith('event: '):
            event_type = line[7:].strip()
        elif line.startswith('data: ') and event_type:
            data = line[6:].strip()
            try:
                yield (event_type, json.loads(data))
            except json.JSONDecodeError:
                continue


def convert_to_anthropic_messages(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert OpenAI messages to Anthropic format."""
    system_prompts: List[str] = []
    anthropic_messages: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str):
                system_prompts.append(content)
        elif role in ["user", "assistant"]:
            anthropic_msg: Dict[str, Any] = {"role": role, "content": []}

            if isinstance(content, str):
                anthropic_msg["content"] = [{"type": "text", "text": content}]
            else:
                # Handle multimodal content
                for part in content:
                    if part["type"] == "text":
                        anthropic_msg["content"].append({
                            "type": "text",
                            "text": part["text"]
                        })
                    elif part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:image/"):
                            # Extract base64 data from data URL
                            media_type = url.split(';')[0].split(':')[1]
                            base64_data = url.split(',')[1]
                            anthropic_msg["content"].append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_data
                                }
                            })
                        else:
                            anthropic_msg["content"].append({
                                "type": "image",
                                "source": {"type": "url", "url": url}
                            })

            anthropic_messages.append(anthropic_msg)

    result: Dict[str, Any] = {"messages": anthropic_messages}
    if system_prompts:
        result["system"] = "\n\n".join(system_prompts)

    return result


def convert_anthropic_to_openai_response(anthropic_response: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    """Convert Anthropic response to OpenAI format."""
    content_parts = []

    for block in anthropic_response.get("content", []):
        if block["type"] == "text":
            content_parts.append(block["text"])

    content = "".join(content_parts)

    # Build OpenAI response
    return {
        "id": anthropic_response.get("id", ""),
        "object": "chat.completion",
        "created": 0,  # Anthropic doesn't provide timestamps
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": anthropic_response.get("stop_reason", "stop")
        }],
        "usage": {
            "prompt_tokens": anthropic_response.get("usage", {}).get("input_tokens", 0),
            "completion_tokens": anthropic_response.get("usage", {}).get("output_tokens", 0),
            "total_tokens": (
                anthropic_response.get("usage", {}).get("input_tokens", 0) +
                anthropic_response.get("usage", {}).get("output_tokens", 0)
            )
        }
    }


def convert_anthropic_stream_to_openai(event_type: str, event_data: Dict[str, Any], model_id: str) -> Optional[Dict[str, Any]]:
    """Convert Anthropic streaming event to OpenAI format."""
    if event_type == "content_block_delta":
        delta = event_data.get("delta", {})
        if delta.get("type") == "text_delta":
            return {
                "id": "",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": delta.get("text", "")
                    },
                    "finish_reason": None
                }]
            }
    elif event_type == "message_stop":
        return {
            "id": "",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }

    return None
