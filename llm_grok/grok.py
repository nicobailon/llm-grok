"""Main Grok model implementation for LLM CLI integration."""
import sys
import threading
from collections.abc import Iterator
from typing import Literal, Optional, Union, cast, TYPE_CHECKING, Protocol, runtime_checkable, TypeGuard, Dict, Any

import httpx
import llm
from pydantic import Field
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

try:
    from llm import KeyModel
    from llm import Options as LLMOptionsBase
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLMOptionsBase = object  # Simple fallback

from .client import GrokClient
from .exceptions import (
    APIError,
    AuthenticationError,
    GrokError,
    NetworkError,
    QuotaExceededError,
    RateLimitError,
)
from .formats import AnthropicFormatHandler, OpenAIFormatHandler
from .models import MODEL_INFO, get_model_info_safe
from .processors import ImageProcessor, StreamProcessor, ToolProcessor
from .types import (
    AnthropicToolChoice,
    AnthropicToolDefinition,
    ImageContent,
    LLMModelProtocol,
    LLMOptionsProtocol,
    LLMPromptProtocol,
    Message,
    RequestBody,
    ResponseFormat,
    TextContent,
    ToolCall,
    ToolChoice,
    ToolDefinition,
)


@runtime_checkable
class ResponseProtocol(Protocol):
    """Protocol for LLM response objects."""
    def text(self) -> str: ...


@runtime_checkable  
class ResponseWithContent(Protocol):
    """Protocol for response objects with content attribute."""
    content: str


@runtime_checkable
class HTTPResponse(Protocol):
    """Protocol for HTTP response objects."""
    status_code: int


@runtime_checkable
class HTTPErrorWithResponse(Protocol):
    """Protocol for HTTP errors with response attribute."""
    response: HTTPResponse


def is_response_like(obj: object) -> TypeGuard[ResponseProtocol]:
    """Type guard for response-like objects."""
    return hasattr(obj, 'text') and callable(getattr(obj, 'text'))


def has_text_method(obj: object) -> TypeGuard[ResponseProtocol]:
    """Type guard for objects with text() method."""
    return hasattr(obj, 'text') and callable(getattr(obj, 'text'))


def has_content_attr(obj: object) -> TypeGuard[ResponseWithContent]:
    """Type guard for objects with content attribute."""
    return hasattr(obj, 'content')


def has_response_attr(obj: object) -> TypeGuard[HTTPErrorWithResponse]:
    """Type guard for errors with response attribute and status_code."""
    if not hasattr(obj, 'response'):
        return False
    
    response = getattr(obj, 'response')
    if not hasattr(response, 'status_code'):
        return False
    
    status_code = getattr(response, 'status_code')
    return isinstance(status_code, int)


console = Console()

# Shared connection pool management
_shared_client_pool: Optional[GrokClient] = None
_client_lock = threading.Lock()
_current_api_key: Optional[str] = None


class GrokOptions:
    """Grok-specific options with LLM framework compatibility."""
    
    def __init__(
        self,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        tools: Optional[list[ToolDefinition]] = None,
        tool_choice: Optional[Union[Literal["auto", "none"], ToolChoice]] = None,
        response_format: Optional[ResponseFormat] = None,
        reasoning_effort: Optional[str] = None,
        use_messages_endpoint: bool = False,
    ) -> None:
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.tools = tools
        self.tool_choice = tool_choice
        self.response_format = response_format
        self.reasoning_effort = reasoning_effort
        self.use_messages_endpoint = use_messages_endpoint
        
        # LLMOptionsProtocol compatibility
        self.max_tokens = max_completion_tokens


class GrokMixin:
    """Mixin class containing all Grok implementation."""
    
    can_stream: bool = True
    needs_key: Optional[str] = "grok"
    key_env_var: Optional[str] = "XAI_API_KEY"
    Options = GrokOptions

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._openai_formatter = OpenAIFormatHandler(model_id)
        self._anthropic_formatter = AnthropicFormatHandler(model_id)
        self._image_processor = ImageProcessor(model_id)
        self._tool_processor = ToolProcessor(model_id)
        self._stream_processor = StreamProcessor(model_id)

    def get_key(self, key: Optional[str] = None) -> str:
        """Get API key for authentication."""
        if key:
            return key
        import os
        api_key = os.environ.get(self.key_env_var or "XAI_API_KEY")
        if api_key is None:
            raise ValueError(f"API key not found. Set {self.key_env_var} environment variable")
        return api_key

    def _get_client(self, api_key: str) -> GrokClient:
        """Get the shared GrokClient instance, creating if necessary."""
        global _shared_client_pool, _current_api_key

        with _client_lock:
            if _shared_client_pool is None:
                _shared_client_pool = GrokClient(api_key=api_key)
                _current_api_key = api_key
            elif _current_api_key != api_key:
                _shared_client_pool.close()
                _shared_client_pool = GrokClient(api_key=api_key)
                _current_api_key = api_key

            return _shared_client_pool

    def _get_model_capability(self, capability: str) -> bool:
        """Check if current model supports a specific capability."""
        model_info = get_model_info_safe(self.model_id)
        return bool(model_info.get(capability, False))

    def _build_message_content(self, prompt: llm.Prompt) -> Union[str, list[Union[TextContent, ImageContent]]]:
        """Build message content, handling multimodal inputs for vision-capable models."""
        supports_vision = self._get_model_capability("supports_vision")
        
        if supports_vision and hasattr(prompt, 'attachments') and prompt.attachments:
            return self._image_processor.process_prompt_with_attachments(prompt)
        else:
            if hasattr(prompt, 'prompt'):
                return self._validate_and_convert_content_list(prompt.prompt)
            else:
                return str(prompt)

    def _validate_and_convert_content_list(
        self, content: Union[str, list[Union[TextContent, ImageContent]]]
    ) -> Union[str, list[Union[TextContent, ImageContent]]]:
        """Validate and convert content list format."""
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            for i, item in enumerate(content):
                if isinstance(item, dict):
                    if "type" not in item:
                        raise ValueError(f"Content item {i} missing 'type' field")
                    if item["type"] not in ["text", "image_url"]:
                        raise ValueError(f"Content item {i} has invalid type: {item['type']}")
                    
                    if item["type"] == "text" and "text" not in item:
                        raise ValueError(f"Text content item {i} missing 'text' field")
                    if item["type"] == "image_url" and "image_url" not in item:
                        raise ValueError(f"Image content item {i} missing 'image_url' field")
                else:
                    raise ValueError(f"Content item {i} must be a dictionary")
            
            return content
        
        raise ValueError("Content must be either string or list of content items")

    def _convert_tools_for_anthropic(
        self, tools: Optional[list[ToolDefinition]]
    ) -> Optional[list[AnthropicToolDefinition]]:
        """Convert OpenAI tool format to Anthropic format if needed."""
        if not tools:
            return None
        return self._openai_formatter.convert_tools_to_anthropic(tools)

    def _convert_tool_choice_for_anthropic(
        self, tool_choice: Optional[Union[Literal["auto", "none"], ToolChoice]]
    ) -> Optional[Union[Literal["auto", "none"], AnthropicToolChoice]]:
        """Convert OpenAI tool choice format to Anthropic format if needed."""
        if not tool_choice:
            return None
        return self._openai_formatter.convert_tool_choice_to_anthropic(tool_choice)

    def build_messages(self, prompt: llm.Prompt, conversation: Optional[llm.Conversation]) -> list[Message]:
        """Build message array for API request."""
        messages: list[Message] = []
        
        if conversation:
            for prev_response in conversation.responses:
                messages.append({
                    "role": "user",
                    "content": prev_response.prompt.prompt
                })
                
                # Safely get response text using type guards
                response_text = ""
                if has_text_method(prev_response):
                    response_text = prev_response.text()
                elif has_content_attr(prev_response):
                    response_text = str(prev_response.content)
                else:
                    response_text = str(prev_response)
                
                messages.append({
                    "role": "assistant", 
                    "content": response_text
                })
        
        content = self._build_message_content(prompt)
        messages.append({
            "role": "user",
            "content": content
        })
        
        return messages

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: Optional[llm.Conversation] = None,
        **options: Any
    ) -> Iterator[str]:
        """Execute model with given prompt and options."""
        try:
            api_key = self.get_key()
            client = self._get_client(api_key)
            messages = self.build_messages(prompt, conversation)
            
            body: Dict[str, Any] = {
                "model": self.model_id,
                "messages": messages,
                "stream": stream
            }
            
            grok_options = GrokOptions()
            # Extract options from kwargs dictionary
            if 'temperature' in options:
                grok_options.temperature = options['temperature']
            if 'max_completion_tokens' in options:
                grok_options.max_completion_tokens = options['max_completion_tokens']
            if 'tools' in options:
                grok_options.tools = options['tools']
            if 'tool_choice' in options:
                grok_options.tool_choice = options['tool_choice']
            if 'response_format' in options:
                grok_options.response_format = options['response_format']
            if 'reasoning_effort' in options:
                grok_options.reasoning_effort = options['reasoning_effort']
            if 'use_messages_endpoint' in options:
                grok_options.use_messages_endpoint = options['use_messages_endpoint']

            if grok_options.temperature is not None:
                body["temperature"] = grok_options.temperature
            if grok_options.max_completion_tokens is not None:
                body["max_tokens"] = grok_options.max_completion_tokens
            if grok_options.response_format is not None:
                body["response_format"] = grok_options.response_format
            if grok_options.reasoning_effort is not None:
                body["reasoning_effort"] = grok_options.reasoning_effort

            if self._get_model_capability("supports_tools") and grok_options.tools:
                # Use tools directly for now - format conversion will be handled later
                body["tools"] = grok_options.tools
                if grok_options.tool_choice:
                    body["tool_choice"] = grok_options.tool_choice

            if grok_options.use_messages_endpoint:
                endpoint_url = "https://api.x.ai/v1/messages"
                formatted_body = body  # Use body directly for now
            else:
                endpoint_url = "https://api.x.ai/v1/chat/completions"
                formatted_body = body  # Use body directly for now

            if stream:
                yield from self._handle_streaming_response(
                    client, endpoint_url, formatted_body, response, grok_options
                )
            else:
                yield from self._handle_non_streaming_response(
                    client, endpoint_url, formatted_body, response, grok_options
                )
                
        except Exception as e:
            self._handle_error(e, "execution")
            raise

    def _handle_streaming_response(
        self,
        client: GrokClient,
        endpoint_url: str,
        body: Dict[str, Any],
        response: llm.Response,
        options: GrokOptions
    ) -> Iterator[str]:
        """Handle streaming response from API."""
        try:
            # Use the client's request method directly
            with client.request(
                method="POST",
                url=endpoint_url,
                json=body,
                stream=True
            ) as http_response:
                for chunk_text in self._stream_processor.process_stream(http_response, response, options.use_messages_endpoint):
                    if chunk_text:
                        yield chunk_text
        
        except Exception as e:
            self._handle_error(e, "streaming")
            raise

    def _handle_non_streaming_response(
        self,
        client: GrokClient,
        endpoint_url: str,
        body: Dict[str, Any],
        response: llm.Response,
        options: GrokOptions
    ) -> Iterator[str]:
        """Handle non-streaming response from API."""
        try:
            # Use the client's request method directly
            http_response = client.request(
                method="POST",
                url=endpoint_url,
                json=body,
                stream=False
            )
            
            api_response = http_response.json()
            
            # Extract content based on format
            if options.use_messages_endpoint:
                # Anthropic format
                if "content" in api_response:
                    content = api_response["content"]
                    if isinstance(content, list):
                        # Extract text from content blocks
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                        text_response = "".join(text_parts)
                    else:
                        text_response = str(content)
                    
                    if text_response:
                        yield text_response
            else:
                # OpenAI format
                if "choices" in api_response and api_response["choices"]:
                    choice = api_response["choices"][0]
                    message = choice.get("message", {})
                    content = message.get("content", "")
                    
                    if content:
                        yield content
        
        except Exception as e:
            self._handle_error(e, "non-streaming")
            raise

    def _handle_error(self, error: Exception, error_type: str) -> None:
        """Handle and log errors with user-friendly messages."""
        if isinstance(error, (NetworkError, httpx.NetworkError, httpx.ConnectError)):
            error_panel = Panel(
                "[red]Network connection failed[/red]\n\n"
                "Please check your internet connection and try again.\n"
                f"Error details: {str(error)}",
                title="Connection Error",
                border_style="red"
            )
        elif isinstance(error, (AuthenticationError, httpx.HTTPStatusError)) and has_response_attr(error) and error.response.status_code == 401:
            error_panel = Panel(
                "[red]Authentication failed[/red]\n\n"
                "Please check your xAI API key. You can:\n"
                "1. Set the XAI_API_KEY environment variable\n"
                "2. Use: llm keys set grok <your-api-key>\n\n"
                "Get your API key from: https://x.ai",
                title="Authentication Error", 
                border_style="red"
            )
        elif isinstance(error, (QuotaExceededError, RateLimitError)):
            error_panel = Panel(
                "[yellow]Rate limit or quota exceeded[/yellow]\n\n"
                "Please wait a moment and try again.\n"
                "Consider upgrading your xAI plan if this persists.",
                title="Rate Limit",
                border_style="yellow"
            )
        else:
            error_panel = Panel(
                f"[red]An error occurred during {error_type}[/red]\n\n"
                f"Error: {str(error)}\n"
                f"Type: {type(error).__name__}",
                title="Error",
                border_style="red"
            )

        if hasattr(sys, 'ps1') or sys.stderr.isatty():
            rprint(error_panel)

        raise


# Always use GrokMixin for implementation regardless of LLM availability
class Grok(GrokMixin):
    """Grok model implementation."""
    pass


def cleanup_shared_resources() -> None:
    """Clean up shared resources including the connection pool."""
    global _shared_client_pool, _current_api_key

    with _client_lock:
        if _shared_client_pool is not None:
            _shared_client_pool.close()
            _shared_client_pool = None
            _current_api_key = None