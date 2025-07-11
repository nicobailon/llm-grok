"""
LLM-Grok Consolidated Implementation - Educational Version

This file provides a complete, single-file implementation of the llm-grok
plugin for educational and reference purposes. For production use, prefer
the modular implementation in the main llm_grok package.

Features: All core functionality in ~500 lines
Limitations: Simplified error handling and no enterprise features

llm-grok plugin for xAI's Grok models.

A clean, efficient implementation that provides access to all Grok models
through the LLM CLI interface.
"""

import base64
import json
import time
from collections.abc import Iterator
from typing import Any, Dict, List, Optional

import httpx
import llm
from pydantic import Field

try:
    from llm_grok.models import AVAILABLE_MODELS as MODELS, DEFAULT_MODEL
except ImportError:
    # Fallback for when used as standalone educational example
    pass

# ============================================================================
# Model Registry - Educational Fallback
# ============================================================================

# For educational/standalone use when models module is not available
if 'MODELS' not in locals():
    MODELS = {
        # Grok 4 models
        "x-ai/grok-4": {
            "context_window": 256000,
            "supports_vision": True,
            "supports_tools": True,
        },
        "grok-4-heavy": {
            "context_window": 256000,
            "supports_vision": True,
            "supports_tools": True,
        },
        # Grok 3 models
        "grok-3-latest": {
            "context_window": 128000,
            "supports_vision": False,
            "supports_tools": False,
        },
        "grok-3-fast-latest": {
            "context_window": 128000,
            "supports_vision": False,
            "supports_tools": False,
        },
        # Grok 2 models
        "grok-2-latest": {
            "context_window": 32768,
            "supports_vision": False,
            "supports_tools": False,
        },
        "grok-2-vision-latest": {
            "context_window": 32768,
            "supports_vision": True,
            "supports_tools": False,
        },
    }
    DEFAULT_MODEL = "x-ai/grok-4"


# ============================================================================
# Exceptions
# ============================================================================

class GrokError(Exception):
    """Base exception for Grok API errors."""
    pass


class RateLimitError(GrokError):
    """Exception raised when API rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        super().__init__(message)
        self.retry_after = retry_after


class APIError(GrokError):
    """General API error for all other failures."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


# ============================================================================
# HTTP Client
# ============================================================================

class GrokClient:
    """HTTP client for Grok API with retry logic."""

    def __init__(self, api_key: str, base_url: str = "https://api.x.ai/v1/chat/completions"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def post(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request to the API with simple retry logic."""
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                with httpx.Client() as client:
                    response = client.post(
                        self.base_url,
                        headers=self.headers,
                        json=data,
                        timeout=60.0
                    )

                    if response.status_code == 200:
                        return response.json()
                    elif response.status_code == 429:
                        # Rate limit
                        retry_after = int(response.headers.get("Retry-After", retry_delay))
                        if attempt < max_retries - 1:
                            time.sleep(retry_after)
                            continue
                        raise RateLimitError(retry_after=retry_after)
                    elif response.status_code == 401:
                        raise APIError("Authentication failed - check your API key", 401)
                    else:
                        error_msg = f"API error: {response.status_code}"
                        try:
                            error_data = response.json()
                            if "error" in error_data:
                                error_msg = error_data["error"].get("message", error_msg)
                        except Exception:
                            pass
                        raise APIError(error_msg, response.status_code)

            except httpx.RequestError as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                raise GrokError(f"Network error: {str(e)}")

        raise GrokError("Max retries exceeded")

    def stream(self, data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        """Make a streaming request to the API."""
        data["stream"] = True

        try:
            with httpx.Client() as client:
                with client.stream(
                    "POST",
                    self.base_url,
                    headers=self.headers,
                    json=data,
                    timeout=60.0
                ) as response:
                    if response.status_code != 200:
                        # Handle error response
                        error_content = response.read()
                        try:
                            error_data = json.loads(error_content)
                            error_msg = error_data.get("error", {}).get("message", f"API error: {response.status_code}")
                        except Exception:
                            error_msg = f"API error: {response.status_code}"

                        if response.status_code == 429:
                            raise RateLimitError()
                        elif response.status_code == 401:
                            raise APIError("Authentication failed", 401)
                        else:
                            raise APIError(error_msg, response.status_code)

                    # Process SSE stream
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                yield json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

        except httpx.RequestError as e:
            raise GrokError(f"Network error: {str(e)}")


# ============================================================================
# Main Model Class
# ============================================================================

class Grok(llm.KeyModel):
    """Grok model implementation for LLM CLI."""

    can_stream = True
    needs_key = "grok"
    key_env_var = "XAI_API_KEY"

    class Options(llm.Options):  # type: ignore[override]
        temperature: Optional[float] = Field(
            description="Sampling temperature (0-1). Lower = more focused, higher = more random.",
            ge=0,
            le=1,
            default=0.7,
        )
        max_completion_tokens: Optional[int] = Field(
            description="Maximum tokens to generate",
            ge=0,
            default=None,
        )
        tools: Optional[List[Dict[str, Any]]] = Field(
            description="List of tool/function definitions",
            default=None,
        )
        tool_choice: Optional[str] = Field(
            description="Tool choice: 'auto', 'none', or specific function name",
            default=None,
        )

    def __init__(self, model_id: str):
        self.model_id = model_id
        # Check if model exists
        if model_id not in MODELS:
            raise ValueError(f"Unknown model: {model_id}")

    def build_messages(self, prompt: llm.Prompt, conversation: Optional[llm.Conversation]) -> List[Dict[str, Any]]:
        """Build messages array from prompt and conversation."""
        messages = []

        # Add conversation history
        if conversation:
            for response in conversation.responses:
                # Add user prompt
                messages.append({"role": "user", "content": response.prompt.prompt or ""})
                # Add assistant response
                # response.text is a property on llm.Response
                messages.append({"role": "assistant", "content": response.text})  # type: ignore[attr-defined]

        # Add system prompt if provided
        if prompt.system:
            messages.insert(0, {"role": "system", "content": prompt.system})

        # Add current prompt with attachments if any
        if hasattr(prompt, 'attachments') and prompt.attachments:
            # Handle multimodal content
            content = []
            if prompt.prompt:
                content.append({"type": "text", "text": prompt.prompt})

            model_info = MODELS.get(self.model_id, {})
            if model_info.get("supports_vision", False):
                for attachment in prompt.attachments:
                    if attachment.resolve_type() == "image":
                        # Handle image attachment
                        image_url = self._process_image(attachment)
                        if image_url:
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            })

            messages.append({"role": "user", "content": content})
        else:
            # Text-only prompt
            messages.append({"role": "user", "content": prompt.prompt})

        return messages

    def _process_image(self, attachment: llm.Attachment) -> Optional[str]:
        """Process image attachment and return URL or base64 data."""
        try:
            # Use content, path, or url depending on what's available
            if attachment.content:
                data = attachment.content
            elif attachment.path:
                data = attachment.path
            elif attachment.url:
                data = attachment.url
            else:
                return None

            # If it's already a URL, return as-is
            if isinstance(data, str) and (data.startswith("http://") or data.startswith("https://")):
                return data

            # If it's base64 data, ensure it has proper data URL format
            if isinstance(data, str) and data.startswith("data:image/"):
                return data

            # If it's raw image bytes, convert to base64
            if isinstance(data, bytes):
                base64_data = base64.b64encode(data).decode('utf-8')
                # Try to detect MIME type from first few bytes
                if data.startswith(b'\xff\xd8\xff'):
                    mime_type = "image/jpeg"
                elif data.startswith(b'\x89PNG'):
                    mime_type = "image/png"
                elif data.startswith(b'GIF8'):
                    mime_type = "image/gif"
                elif data.startswith(b'RIFF') and b'WEBP' in data[:12]:
                    mime_type = "image/webp"
                else:
                    mime_type = "image/png"  # Default

                return f"data:{mime_type};base64,{base64_data}"

            # Try to read as file path
            try:
                with open(data, 'rb') as f:
                    image_bytes = f.read()
                    base64_data = base64.b64encode(image_bytes).decode('utf-8')
                    return f"data:image/png;base64,{base64_data}"
            except Exception:
                pass

            return None
        except Exception:
            return None

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: Optional[llm.Conversation],
        key: Optional[str] = None,
    ) -> Iterator[str]:
        """Execute a prompt against the model."""
        try:
            messages = self.build_messages(prompt, conversation)

            # Get options from prompt
            options = prompt.options.__dict__ if prompt.options else {}

            # Build request body
            body = {
                "model": self.model_id,
                "messages": messages,
                "temperature": options.get("temperature", 0.7),
            }

            if options.get("max_completion_tokens"):
                body["max_completion_tokens"] = options["max_completion_tokens"]

            # Add tools if supported by model
            model_info = MODELS.get(self.model_id, {})
            if model_info.get("supports_tools", False) and options.get("tools"):
                body["tools"] = options["tools"]
                if options.get("tool_choice"):
                    body["tool_choice"] = options["tool_choice"]

            # Get API key
            api_key = key or self.key
            if not api_key:
                raise APIError("No API key found. Set XAI_API_KEY environment variable.", 401)

            # Create client
            client = GrokClient(api_key)

            if stream:
                # Streaming response - yield text chunks
                for chunk_text in self._stream(client, body):
                    yield chunk_text
            else:
                # Non-streaming request
                api_response = client.post(body)

                # Store response JSON for later access
                response.response_json = api_response

                # Extract content and tool calls
                content = ""
                tool_calls = None

                if "choices" in api_response and api_response["choices"]:
                    choice = api_response["choices"][0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    tool_calls = message.get("tool_calls")

                # Add tool calls to response if present
                if tool_calls:
                    for tool_call in tool_calls:
                        import json
                        arguments_str = tool_call.get("function", {}).get("arguments", "{}")
                        try:
                            arguments_dict = json.loads(arguments_str)
                        except json.JSONDecodeError:
                            arguments_dict = {}
                        response.add_tool_call(llm.ToolCall(
                            tool_call_id=tool_call.get("id"),
                            name=tool_call.get("function", {}).get("name"),
                            arguments=arguments_dict
                        ))

                # Yield the content
                if content:
                    yield content

        except RateLimitError as e:
            raise llm.ModelError(f"Rate limit exceeded{f': retry after {e.retry_after}s' if e.retry_after else ''}")
        except APIError as e:
            if e.status_code == 401:
                raise llm.NeedsKeyException("Invalid API key")
            raise llm.ModelError(str(e))
        except Exception as e:
            raise llm.ModelError(f"Error: {str(e)}")

    def _stream(self, client: GrokClient, body: Dict[str, Any]) -> Iterator[str]:
        """Handle streaming responses."""
        try:
            for chunk in client.stream(body):
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        yield delta["content"]
        except Exception as e:
            raise llm.ModelError(f"Streaming error: {str(e)}")


# ============================================================================
# Plugin Registration - Educational Version
# ============================================================================

# NOTE: This is an educational implementation. For production use, the
# plugin registration is handled by the main llm_grok package.
# These functions are commented out to prevent conflicts when used as an example.

# @llm.hookimpl
def register_models_educational(register) -> None:
    """Register Grok models with LLM - Educational version."""
    # Register all models
    for model_id in MODELS:
        register(Grok(model_id))

    # Register default model alias
    if DEFAULT_MODEL.startswith("x-ai/"):
        short_name = DEFAULT_MODEL[5:]  # Remove "x-ai/" prefix
        register(Grok(DEFAULT_MODEL), aliases=[short_name])


# ============================================================================
# CLI Commands - Educational Version
# ============================================================================

# @llm.hookimpl
def register_commands_educational(cli) -> None:
    """Register CLI commands - Educational version."""
    import click

    @cli.group()
    def grok() -> None:
        """Commands for working with Grok models"""
        pass

    @grok.command()  # type: ignore[misc]
    def models() -> None:
        """List available Grok models"""
        click.echo("Available Grok models:\n")
        for model_id, info in MODELS.items():
            click.echo(f"  {model_id}")
            click.echo(f"    Context window: {info['context_window']:,} tokens")
            click.echo(f"    Vision support: {'Yes' if info['supports_vision'] else 'No'}")
            click.echo(f"    Tool support: {'Yes' if info['supports_tools'] else 'No'}")
            click.echo()


# ============================================================================
# Backwards Compatibility
# ============================================================================

# For backwards compatibility with existing imports
__version__ = "3.0.0"
__all__ = ["Grok", "register_models", "register_commands"]
