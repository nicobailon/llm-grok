"""Main Grok model implementation for LLM CLI integration."""
import sys
import threading
from collections.abc import Iterator
from typing import Literal, Optional, Union, cast, TYPE_CHECKING, Protocol, runtime_checkable, TypeGuard, Dict, Any, ClassVar
from contextlib import AbstractContextManager

import httpx
import llm
from pydantic import Field
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    from llm import KeyModel as LLMKeyModel
    from llm.models import Options as LLMOptionsBase
    LLM_AVAILABLE = True
else:
    try:
        from llm import KeyModel as LLMKeyModel
        from llm.models import Options as LLMOptionsBase
        LLM_AVAILABLE = True
    except ImportError:
        LLM_AVAILABLE = False
        # Create a proper base class for fallback
        class LLMOptionsBase:
            pass
        class LLMKeyModel:
            pass

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
    AnthropicRequest,
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


class GrokOptions(LLMOptionsBase):
    """Grok-specific options with LLM framework compatibility."""
    
    temperature: Optional[float] = Field(default=0.0)
    max_completion_tokens: Optional[int] = Field(default=None)
    tools: Optional[list[ToolDefinition]] = Field(default=None)
    tool_choice: Optional[Union[Literal["auto", "none"], ToolChoice]] = Field(default=None)
    response_format: Optional[ResponseFormat] = Field(default=None)
    reasoning_effort: Optional[str] = Field(default=None)
    use_messages_endpoint: bool = Field(default=False)
    max_tokens: Optional[int] = Field(default=None)  # LLMOptionsProtocol compatibility
    
    def model_post_init(self, __context: Any) -> None:
        # LLMOptionsProtocol compatibility - sync max_tokens with max_completion_tokens
        if self.max_completion_tokens is not None:
            object.__setattr__(self, 'max_tokens', self.max_completion_tokens)


class GrokMixin(LLMKeyModel):
    """Mixin class containing all Grok implementation."""
    
    can_stream: bool = True
    needs_key: Optional[str] = "grok"
    key_env_var: Optional[str] = "XAI_API_KEY"
    
    # Cast to the actual expected type from the LLM framework
    Options = cast("type[llm.KeyModel.Options]", GrokOptions)

    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.model_id = model_id
        self._openai_formatter = OpenAIFormatHandler(model_id)
        self._anthropic_formatter = AnthropicFormatHandler(model_id)
        self._image_processor = ImageProcessor(model_id)
        self._tool_processor = ToolProcessor(model_id)
        self._stream_processor = StreamProcessor(model_id)

    def get_key(self, explicit_key: Optional[str] = None) -> Optional[str]:
        """Get API key for authentication."""
        if explicit_key:
            return explicit_key
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
        
        # Add system message if present
        if hasattr(prompt, 'system') and prompt.system:
            messages.append({
                "role": "system",
                "content": prompt.system
            })
        
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
        key: Optional[str] = None
    ) -> Iterator[str]:
        """Execute model with given prompt and options."""
        try:
            api_key = self.get_key(key)
            if api_key is None:
                raise ValueError(f"API key not found. Set {self.key_env_var} environment variable")
            client = self._get_client(api_key)
            messages = self.build_messages(prompt, conversation)
            
            body: Dict[str, Any] = {
                "model": self.model_id,
                "messages": messages,
                "stream": stream
            }
            
            grok_options = GrokOptions()
            # Extract options from prompt.options using getattr for type safety
            options = prompt.options
            if options:
                temperature = getattr(options, 'temperature', None)
                if temperature is not None:
                    grok_options.temperature = temperature
                
                max_completion_tokens = getattr(options, 'max_completion_tokens', None)
                if max_completion_tokens is not None:
                    grok_options.max_completion_tokens = max_completion_tokens
                
                tools = getattr(options, 'tools', None)
                if tools is not None:
                    grok_options.tools = tools
                
                tool_choice = getattr(options, 'tool_choice', None)
                if tool_choice is not None:
                    grok_options.tool_choice = tool_choice
                
                response_format = getattr(options, 'response_format', None)
                if response_format is not None:
                    grok_options.response_format = response_format
                
                reasoning_effort = getattr(options, 'reasoning_effort', None)
                if reasoning_effort is not None:
                    grok_options.reasoning_effort = reasoning_effort
                
                use_messages_endpoint = getattr(options, 'use_messages_endpoint', None)
                if use_messages_endpoint is not None:
                    grok_options.use_messages_endpoint = use_messages_endpoint

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
            # Use the appropriate high-level client method for streaming
            if options.use_messages_endpoint:
                # Convert to Anthropic format and pass as request_data
                anthropic_request_dict = self._anthropic_formatter.convert_from_openai(body)
                anthropic_request = cast(AnthropicRequest, anthropic_request_dict)
                http_response = client.post_anthropic_messages(
                    request_data=anthropic_request,
                    model=self.model_id,
                    stream=True
                )
            else:
                # Extract messages for OpenAI endpoint
                messages = body.get("messages", [])
                http_response = client.post_openai_completion(
                    messages=messages,
                    model=self.model_id,
                    stream=True,
                    temperature=body.get("temperature", 0.7),
                    max_completion_tokens=body.get("max_tokens"),
                    tools=body.get("tools"),
                    tool_choice=body.get("tool_choice"),
                    response_format=body.get("response_format"),
                    reasoning_effort=body.get("reasoning_effort")
                )
            
            # Type narrowing: streaming responses are context managers
            if isinstance(http_response, AbstractContextManager):
                with http_response as response_stream:
                    for chunk_text in self._stream_processor.process_stream(response_stream, response, options.use_messages_endpoint):
                        if chunk_text:
                            yield chunk_text
            else:
                # Fallback for non-context manager responses
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
            # Use the appropriate high-level client method for non-streaming
            if options.use_messages_endpoint:
                # Convert to Anthropic format and pass as request_data
                anthropic_request_dict = self._anthropic_formatter.convert_from_openai(body)
                anthropic_request = cast(AnthropicRequest, anthropic_request_dict)
                http_response = client.post_anthropic_messages(
                    request_data=anthropic_request,
                    model=self.model_id,
                    stream=False
                )
            else:
                # Extract messages for OpenAI endpoint
                messages = body.get("messages", [])
                http_response = client.post_openai_completion(
                    messages=messages,
                    model=self.model_id,
                    stream=False,
                    temperature=body.get("temperature", 0.7),
                    max_completion_tokens=body.get("max_tokens"),
                    tools=body.get("tools"),
                    tool_choice=body.get("tool_choice"),
                    response_format=body.get("response_format"),
                    reasoning_effort=body.get("reasoning_effort")
                )
            
            # Type narrowing: non-streaming responses have json() method
            if hasattr(http_response, 'json'):
                api_response = http_response.json()
            else:
                raise ValueError("Invalid response type from client")
            
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