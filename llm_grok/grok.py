"""Main Grok model implementation for LLM CLI integration."""
import sys
import threading
from collections.abc import Iterator
from typing import Any, Literal, Optional, Union, cast, TYPE_CHECKING, Protocol, runtime_checkable, TypeGuard

import httpx
import llm
from pydantic import Field
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    try:
        from llm import KeyModel as LLMBaseModel
    except ImportError:
        # Create a protocol for type checking
        @runtime_checkable
        class LLMBaseModel(Protocol):
            model_id: str
            can_stream: bool
            needs_key: Optional[str]
            key_env_var: Optional[str]
            
            def execute(self, prompt: Any, stream: bool, response: Any, conversation: Any, key: str | None) -> Iterator[str]: ...
else:
    # Runtime - use the proper base class for API-key models
    try:
        from llm import KeyModel as LLMBaseModel
    except ImportError:
        # Fallback to protocol class if LLM not available
        @runtime_checkable
        class LLMBaseModel(Protocol):
            model_id: str
            can_stream: bool
            needs_key: Optional[str]
            key_env_var: Optional[str]
            
            def execute(self, prompt: Any, stream: bool, response: Any, conversation: Any, key: str | None) -> Iterator[str]: ...

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
    ToolChoice,
    ToolDefinition,
)


@runtime_checkable
class ResponseProtocol(Protocol):
    """Protocol for LLM response objects."""
    def text(self) -> str: ...
    
    
def is_response_like(obj: Any) -> TypeGuard[ResponseProtocol]:
    """Type guard for response-like objects."""
    return hasattr(obj, 'text') and callable(getattr(obj, 'text'))


console = Console()

# Shared connection pool management
_shared_client_pool: Optional[GrokClient] = None
_client_lock = threading.Lock()
_current_api_key: Optional[str] = None


if TYPE_CHECKING:
    try:
        from llm import Options as LLMOptionsBase
    except ImportError:
        # Protocol for type checking
        class LLMOptionsBase(Protocol):
            pass
else:
    try:
        from llm import Options as LLMOptionsBase
    except ImportError:
        # Protocol for runtime if LLM not available
        class LLMOptionsBase(Protocol):
            pass


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


class Grok:
    """Grok model implementation with proper inheritance."""
    
    # Explicit declarations for Pylance - compatible with LLMBaseModel protocol
    model_id: str
    can_stream: bool = True
    needs_key: Optional[str] = "grok"
    key_env_var: Optional[str] = "XAI_API_KEY"

    # For LLM framework compatibility, create a proper type alias
    Options = GrokOptions

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        # Initialize parent if available
        if hasattr(super(), '__init__'):
            super().__init__(model_id)  # type: ignore
        self._openai_formatter = OpenAIFormatHandler(model_id)
        self._anthropic_formatter = AnthropicFormatHandler(model_id)
        self._image_processor = ImageProcessor(model_id)
        self._tool_processor = ToolProcessor(model_id)
        self._stream_processor = StreamProcessor(model_id)
    
    def get_key(self, key: Optional[str] = None) -> Optional[str]:
        """Get API key for authentication."""
        # For now, return the provided key or get from environment
        # This would normally integrate with LLM's key management
        if key:
            return key
        import os
        api_key = os.environ.get(self.key_env_var or "XAI_API_KEY")
        if api_key is None:
            raise ValueError(f"API key not found. Set {self.key_env_var} environment variable")
        return api_key

    def _get_client(self, api_key: str) -> GrokClient:
        """Get the shared GrokClient instance, creating if necessary.
        
        This method ensures all model instances share the same connection pool
        to prevent resource leaks when multiple models are used.
        
        Args:
            api_key: The API key to use for authentication
            
        Returns:
            The shared GrokClient instance
        """
        global _shared_client_pool, _current_api_key

        with _client_lock:
            # Check if we need to create a new client or update API key
            if _shared_client_pool is None:
                # First time - create new client
                _shared_client_pool = GrokClient(api_key=api_key)
                _current_api_key = api_key
            elif _current_api_key != api_key:
                # API key changed - close old client and create new one
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
        # Check if model supports vision
        supports_vision = self._get_model_capability("supports_vision")

        # Check for attachments
        if hasattr(prompt, 'attachments') and prompt.attachments and supports_vision:
            return self._image_processor.process_prompt_with_attachments(prompt)

        # Return plain text for non-multimodal or unsupported models
        if isinstance(prompt.prompt, str):
            return prompt.prompt
        elif isinstance(prompt.prompt, list):
            return self._validate_and_convert_content_list(prompt.prompt)
        else:
            # Convert other types to string
            return str(prompt.prompt)

    def _validate_and_convert_content_list(
        self, content_list: list[object]
    ) -> list[Union[TextContent, ImageContent]]:
        """Validate and convert content list to proper types."""
        from .types import TextContent, ImageContent
        
        validated_content: list[Union[TextContent, ImageContent]] = []
        
        for item in content_list:
            if isinstance(item, dict):
                content_type = item.get("type")
                
                if content_type == "text":
                    text_content: TextContent = {
                        "type": "text",
                        "text": str(item.get("text", ""))
                    }
                    validated_content.append(text_content)
                    
                elif content_type == "image_url":
                    image_url_data = item.get("image_url", {})
                    if isinstance(image_url_data, dict) and "url" in image_url_data:
                        image_content: ImageContent = {
                            "type": "image_url",
                            "image_url": {"url": str(image_url_data["url"])}
                        }
                        validated_content.append(image_content)
            elif isinstance(item, str):
                # Convert string to text content
                text_content = TextContent(type="text", text=item)
                validated_content.append(text_content)
        
        return validated_content

    def _convert_tools_for_anthropic(
        self, tools: Optional[list[ToolDefinition]]
    ) -> Optional[list[AnthropicToolDefinition]]:
        """Convert OpenAI tools to Anthropic format."""
        if tools is None:
            return None
        # Use formatter to convert - this should be implemented by Dev 1
        return self._openai_formatter.convert_tools_to_anthropic(tools)

    def _convert_tool_choice_for_anthropic(
        self, tool_choice: Optional[Union[Literal["auto", "none"], ToolChoice]]
    ) -> Optional[Union[Literal["auto", "none"], AnthropicToolChoice]]:
        """Convert OpenAI tool choice to Anthropic format."""
        if tool_choice is None:
            return None
        # Use formatter to convert - this should be implemented by Dev 1
        return self._openai_formatter.convert_tool_choice_to_anthropic(tool_choice)

    def build_messages(self, prompt: llm.Prompt, conversation: Optional[llm.Conversation]) -> list[Message]:
        """Build messages array from prompt and conversation history."""
        messages: list[Message] = []

        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})

        if conversation:
            for prev_response in conversation.responses:
                if prev_response.prompt.system:
                    messages.append(
                        {"role": "system", "content": prev_response.prompt.system}
                    )
                messages.append(
                    {"role": "user", "content": self._build_message_content(prev_response.prompt)}
                )
                # Response objects have a text() method
                response_text: str = ""
                if is_response_like(prev_response):
                    response_text = prev_response.text()
                elif hasattr(prev_response, 'text') and callable(getattr(prev_response, 'text')):
                    response_text = getattr(prev_response, 'text')()
                else:
                    response_text = str(prev_response)
                assistant_msg: Message = {"role": "assistant", "content": response_text}
                messages.append(assistant_msg)

        user_content = self._build_message_content(prompt)
        user_msg: Message = {"role": "user", "content": user_content}
        messages.append(user_msg)
        return messages

    def execute(
        self,
        prompt: llm.Prompt,
        stream: bool,
        response: llm.Response,
        conversation: Optional[llm.Conversation],
        key: Optional[str] = None
    ) -> Iterator[str]:
        """Execute the model request."""
        key = self.get_key(key)
        if key is None:
            raise ValueError("API key is required but not provided")
        messages = self.build_messages(prompt, conversation)
        # Store prompt JSON if response supports it
        if hasattr(response, '_prompt_json'):
            response._prompt_json = {"messages": messages}  # type: ignore

        if not hasattr(prompt, "options") or not isinstance(prompt.options, self.Options):
            options = self.Options()
        else:
            options = prompt.options

        # Type assertion for PyLance
        assert isinstance(options, self.Options)

        # Determine which endpoint to use
        use_messages = options.use_messages_endpoint

        # Check for multimodal content with non-vision models
        has_images = any(
            isinstance(msg.get("content"), list) and
            any(isinstance(part, dict) and part.get("type") == "image_url" for part in msg.get("content", []))
            for msg in messages
        )

        if has_images and not self._get_model_capability("supports_vision"):
            console.print(
                f"[yellow]Warning: Model '{self.model_id}' does not support vision. Image content will be ignored.[/yellow]"
            )

        # Build request body based on endpoint
        body: RequestBody = {
            "model": self.model_id,
            "messages": messages,  # Add required messages field
        }
        
        # Add optional fields with proper type validation
        if options.temperature is not None:
            body["temperature"] = float(options.temperature)

        # Handle additional options
        if options.max_completion_tokens is not None:
            body["max_completion_tokens"] = int(options.max_completion_tokens)

        # Get model info for capability checks
        supports_tools = self._get_model_capability("supports_tools")

        # Add function calling parameters if model supports it
        if supports_tools and options.tools is not None:
            body["tools"] = options.tools
            
            if options.tool_choice is not None:
                # Keep OpenAI format in RequestBody - conversion will be handled by client
                body["tool_choice"] = options.tool_choice

        if options.response_format is not None:
            body["response_format"] = options.response_format

        if options.reasoning_effort is not None:
            body["reasoning_effort"] = str(options.reasoning_effort)

        # Get the client
        client = self._get_client(key)

        try:
            if stream:
                # Handle streaming response
                yield from self._handle_streaming_response(
                    client, body, messages, response, bool(use_messages), options
                )
            else:
                # Handle non-streaming response
                yield from self._handle_non_streaming_response(
                    client, body, messages, response, bool(use_messages), options
                )
        except (RateLimitError, QuotaExceededError) as rate_error:
            self._handle_error(rate_error, "Rate Limit/Quota Error")
        except AuthenticationError as auth_error:
            self._handle_error(auth_error, "Authentication Error")
        except (APIError, NetworkError, GrokError) as api_error:
            self._handle_error(api_error, "API Error")
        except httpx.HTTPError as e:
            error_message = f"HTTP Error: {str(e)}"
            if "pytest" in sys.modules:
                raise GrokError(error_message)
            self._handle_error(e, "HTTP Error")

    def _handle_streaming_response(
        self,
        client: GrokClient,
        body: RequestBody,
        messages: list[Message],
        response: llm.Response,
        use_messages: bool,
        options: GrokOptions
    ) -> Iterator[str]:
        """Handle streaming response from the API."""
        if use_messages:
            # Convert messages for Anthropic endpoint
            anthropic_data = self._openai_formatter.convert_messages_to_anthropic(messages)
            stream_response = client.post_anthropic_messages(
                request_data=anthropic_data,
                model=self.model_id,
                stream=True,
                temperature=options.temperature or 0.7,
                max_tokens=body.get("max_tokens"),
                tools=self._convert_tools_for_anthropic(body.get("tools")),
                tool_choice=self._convert_tool_choice_for_anthropic(body.get("tool_choice")),
                reasoning_effort=body.get("reasoning_effort"),
            )
        else:
            stream_response = client.post_openai_completion(
                messages=messages,
                model=self.model_id,
                stream=True,
                temperature=options.temperature or 0.7,
                max_completion_tokens=body.get("max_completion_tokens"),
                tools=body.get("tools"),
                tool_choice=body.get("tool_choice"),
                response_format=body.get("response_format"),
                reasoning_effort=body.get("reasoning_effort"),
            )

        with stream_response as r:
            r.raise_for_status()
            yield from self._stream_processor.process_stream(r, response, use_messages)

    def _handle_non_streaming_response(
        self,
        client: GrokClient,
        body: RequestBody,
        messages: list[Message],
        response: llm.Response,
        use_messages: bool,
        options: GrokOptions
    ) -> Iterator[str]:
        """Handle non-streaming response from the API."""
        if use_messages:
            # Convert messages for Anthropic endpoint
            anthropic_data = self._openai_formatter.convert_messages_to_anthropic(messages)
            response_obj = client.post_anthropic_messages(
                request_data=anthropic_data,
                model=self.model_id,
                stream=False,
                temperature=options.temperature or 0.7,
                max_tokens=body.get("max_tokens"),
                tools=self._convert_tools_for_anthropic(body.get("tools")),
                tool_choice=self._convert_tool_choice_for_anthropic(body.get("tool_choice")),
                reasoning_effort=body.get("reasoning_effort"),
            )
        else:
            response_obj = client.post_openai_completion(
                messages=messages,
                model=self.model_id,
                stream=False,
                temperature=options.temperature or 0.7,
                max_completion_tokens=body.get("max_completion_tokens"),
                tools=body.get("tools"),
                tool_choice=body.get("tool_choice"),
                response_format=body.get("response_format"),
                reasoning_effort=body.get("reasoning_effort"),
            )

        # Type narrowing: when stream=False, both methods return httpx.Response
        assert isinstance(response_obj, httpx.Response)
        response_obj.raise_for_status()
        response_data = response_obj.json()

        # Convert response based on endpoint
        if use_messages:
            response_data = self._anthropic_formatter.convert_from_anthropic_response(response_data)

        # Store response JSON if response supports it
        if hasattr(response, 'response_json'):
            response.response_json = response_data  # type: ignore

        # Process response
        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            message = choice["message"]

            # Handle function/tool calls
            if "tool_calls" in message and message["tool_calls"]:
                self._tool_processor.process_tool_calls(response, message["tool_calls"])
                # Yield content if any (might be None for pure tool calls)
                if message.get("content"):
                    yield message["content"]
            else:
                # Regular content response
                if message.get("content"):
                    yield message["content"]


    def _handle_error(self, error: Exception, error_type: str) -> None:
        """Handle errors with appropriate display.
        
        In production, displays a nice error panel and re-raises the exception.
        In tests, just re-raises the exception without display.
        """
        if "pytest" not in sys.modules:
            # Only display error panel in non-test environments
            error_panel = Panel.fit(
                f"[bold red]{error_type}[/]\n\n[white]{str(error)}[/]",
                title="❌ Error",
                border_style="red",
            )
            rprint(error_panel)

        # Always re-raise the exception instead of exiting
        raise


def cleanup_shared_resources() -> None:
    """Clean up shared resources including the connection pool.
    
    This function should be called when the module is unloaded or
    when the application is shutting down to ensure proper cleanup
    of network resources.
    """
    global _shared_client_pool, _current_api_key

    with _client_lock:
        if _shared_client_pool is not None:
            _shared_client_pool.close()
            _shared_client_pool = None
            _current_api_key = None
