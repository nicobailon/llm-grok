"""Main Grok model implementation for LLM CLI integration."""
import base64
import json
import sys
import threading
import warnings
from typing import Any, Dict, Iterator, List, Optional, Union, cast, Literal

import httpx
import llm
from pydantic import Field
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

from .client import GrokClient
from .constants import IMAGE_FETCH_TIMEOUT, HTTP_CHUNK_SIZE
from .exceptions import (
    APIError,
    AuthenticationError,
    GrokError,
    NetworkError,
    QuotaExceededError,
    RateLimitError,
    ValidationError,
)
from .formats import AnthropicFormatHandler, OpenAIFormatHandler
from .models import MODEL_INFO
from .processors import ImageProcessor, StreamProcessor, ToolProcessor
from .types import (
    ImageContent, Message, TextContent, ToolDefinition,
    ResponseFormat, ToolChoice, RequestBody
)

console = Console()

# Shared connection pool management
_shared_client_pool: Optional[GrokClient] = None
_client_lock = threading.Lock()
_current_api_key: Optional[str] = None


class Grok(llm.KeyModel):
    """Grok model implementation for LLM CLI."""
    
    can_stream = True
    needs_key = "grok"
    key_env_var = "XAI_API_KEY"
    
    class Options(llm.Options):  # type: ignore[assignment]
        temperature: Optional[float] = Field(
            description=(
                "Determines the sampling temperature. Higher values like 0.8 increase randomness, "
                "while lower values like 0.2 make the output more focused and deterministic."
            ),
            ge=0,
            le=1,
            default=0.0,
        )
        max_completion_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate, including visible output tokens and reasoning tokens.",
            ge=0,
            default=None,
        )
        tools: Optional[List[ToolDefinition]] = Field(
            description="List of tool/function definitions in OpenAI format",
            default=None,
        )
        tool_choice: Optional[Union[Literal["auto", "none"], ToolChoice]] = Field(
            description="Controls which (if any) function is called. Can be 'auto', 'none', or a specific function",
            default=None,
        )
        response_format: Optional[ResponseFormat] = Field(
            description="Structured output format (e.g., {'type': 'json_object'})",
            default=None,
        )
        reasoning_effort: Optional[str] = Field(
            description="Level of reasoning effort for the model",
            default=None,
        )
        use_messages_endpoint: Optional[bool] = Field(
            description="Use Anthropic-style /messages endpoint instead of /chat/completions",
            default=False,
        )

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._openai_formatter = OpenAIFormatHandler(model_id)
        self._anthropic_formatter = AnthropicFormatHandler(model_id)
        self._image_processor = ImageProcessor(model_id)
        self._tool_processor = ToolProcessor(model_id)
        self._stream_processor = StreamProcessor(model_id)
    
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
        return MODEL_INFO.get(self.model_id, {}).get(capability, False)
    
    def _build_message_content(self, prompt: llm.Prompt) -> Union[str, List[Union[TextContent, ImageContent]]]:
        """Build message content, handling multimodal inputs for vision-capable models."""
        # Check if model supports vision
        supports_vision = self._get_model_capability("supports_vision")
        
        # Check for attachments
        if hasattr(prompt, 'attachments') and prompt.attachments and supports_vision:
            return self._image_processor.process_prompt_with_attachments(prompt)
        
        # Return plain text for non-multimodal or unsupported models
        return prompt.prompt

    def build_messages(self, prompt: llm.Prompt, conversation: Optional[llm.Conversation]) -> List[Message]:
        """Build messages array from prompt and conversation history."""
        messages: List[Message] = []

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
                if hasattr(prev_response, 'text') and callable(getattr(prev_response, 'text')):
                    response_text = cast(Any, prev_response).text()
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
        setattr(response, '_prompt_json', {"messages": messages})

        if not hasattr(prompt, "options") or not isinstance(prompt.options, self.Options):
            options = self.Options()
        else:
            options = prompt.options
        
        # Type assertion for PyLance
        assert isinstance(options, self.Options)

        # Determine which endpoint to use
        use_messages = options.use_messages_endpoint if options.use_messages_endpoint is not None else False
        
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
            "stream": stream,
            "temperature": options.temperature,
        }
        
        if use_messages:
            # Convert to Anthropic format
            anthropic_data = self._openai_formatter.convert_messages_to_anthropic(messages)
            body["messages"] = anthropic_data.get("messages", [])
            if "system" in anthropic_data:
                body["system"] = anthropic_data["system"]
        else:
            # Standard OpenAI format
            body["messages"] = messages

        # Get model info for capability checks
        supports_tools = self._get_model_capability("supports_tools")

        # Handle max_completion_tokens
        if options.max_completion_tokens is not None:
            model_info = MODEL_INFO.get(self.model_id, {})
            max_output_tokens = model_info.get("max_output_tokens")
            
            if max_output_tokens and options.max_completion_tokens > max_output_tokens:
                console.print(
                    f"[yellow]Warning: max_completion_tokens ({options.max_completion_tokens}) "
                    f"exceeds model's limit ({max_output_tokens}). Clamping to model limit.[/yellow]"
                )
                if use_messages:
                    body["max_tokens"] = max_output_tokens
                else:
                    body["max_completion_tokens"] = max_output_tokens
            else:
                if use_messages:
                    body["max_tokens"] = options.max_completion_tokens
                else:
                    body["max_completion_tokens"] = options.max_completion_tokens
        
        # Add function calling parameters if model supports it
        if supports_tools:
            if options.tools is not None:
                if use_messages:
                    body["tools"] = self._openai_formatter.convert_tools_to_anthropic(options.tools)
                else:
                    body["tools"] = options.tools
                
            if options.tool_choice is not None:
                if use_messages:
                    # Convert tool_choice to Anthropic format
                    if options.tool_choice == "auto":
                        body["tool_choice"] = {"type": "auto"}
                    elif options.tool_choice == "none":
                        body["tool_choice"] = {"type": "none"}
                    elif isinstance(options.tool_choice, dict) and "function" in options.tool_choice:
                        body["tool_choice"] = {
                            "type": "tool",
                            "name": options.tool_choice["function"]["name"]
                        }
                else:
                    body["tool_choice"] = options.tool_choice
                
            if options.response_format is not None:
                if not use_messages:
                    body["response_format"] = options.response_format
                else:
                    console.print("[yellow]Warning: response_format is not supported with messages endpoint[/yellow]")
                
        if options.reasoning_effort is not None:
            body["reasoning_effort"] = options.reasoning_effort

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
        messages: List[Message],
        response: llm.Response,
        use_messages: bool,
        options: Options
    ) -> Iterator[str]:
        """Handle streaming response from the API."""
        if use_messages:
            # Convert messages for Anthropic endpoint
            anthropic_data = self._openai_formatter.convert_messages_to_anthropic(messages)
            stream_response = client.post_anthropic_messages(
                request_data=anthropic_data,
                model=self.model_id,
                stream=True,
                temperature=options.temperature,
                max_tokens=body.get("max_tokens"),
                tools=body.get("tools"),
                tool_choice=body.get("tool_choice"),
                reasoning_effort=body.get("reasoning_effort"),
            )
        else:
            stream_response = client.post_openai_completion(
                messages=messages,
                model=self.model_id,
                stream=True,
                temperature=options.temperature,
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
        messages: List[Message],
        response: llm.Response,
        use_messages: bool,
        options: Options
    ) -> Iterator[str]:
        """Handle non-streaming response from the API."""
        if use_messages:
            # Convert messages for Anthropic endpoint
            anthropic_data = self._openai_formatter.convert_messages_to_anthropic(messages)
            r = client.post_anthropic_messages(
                request_data=anthropic_data,
                model=self.model_id,
                stream=False,
                temperature=options.temperature,
                max_tokens=body.get("max_tokens"),
                tools=body.get("tools"),
                tool_choice=body.get("tool_choice"),
                reasoning_effort=body.get("reasoning_effort"),
            )
        else:
            r = client.post_openai_completion(
                messages=messages,
                model=self.model_id,
                stream=False,
                temperature=options.temperature,
                max_completion_tokens=body.get("max_completion_tokens"),
                tools=body.get("tools"),
                tool_choice=body.get("tool_choice"),
                response_format=body.get("response_format"),
                reasoning_effort=body.get("reasoning_effort"),
            )
        
        r.raise_for_status()
        response_data = r.json()
        
        # Convert response based on endpoint
        if use_messages:
            response_data = self._anthropic_formatter.convert_from_anthropic_response(response_data)
        
        response.response_json = response_data
        
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

    def _validate_image_format(self, image_data: str) -> str:
        """Validate and format image data for API consumption.
        
        Thin wrapper around ImageProcessor.validate_image_format for backward compatibility.
        
        .. deprecated:: 3.0
            Use self._image_processor.validate_image_format() directly instead.
        """
        warnings.warn(
            "_validate_image_format is deprecated and will be removed in v4.0. "
            "Use self._image_processor.validate_image_format() directly.",
            DeprecationWarning,
            stacklevel=2
        )
        return self._image_processor.validate_image_format(image_data)
    
    def _convert_to_anthropic_messages(self, openai_messages: List[Message]) -> Dict[str, Any]:
        """Convert OpenAI message format to Anthropic format.
        
        Thin wrapper around OpenAIFormatHandler.convert_messages_to_anthropic for backward compatibility.
        
        .. deprecated:: 3.0
            Use self._openai_formatter.convert_messages_to_anthropic() directly instead.
        """
        warnings.warn(
            "_convert_to_anthropic_messages is deprecated and will be removed in v4.0. "
            "Use self._openai_formatter.convert_messages_to_anthropic() directly.",
            DeprecationWarning,
            stacklevel=2
        )
        return self._openai_formatter.convert_messages_to_anthropic(openai_messages)
    
    def _convert_tools_to_anthropic(self, openai_tools: List[ToolDefinition]) -> List[Dict[str, Any]]:
        """Convert OpenAI tool definitions to Anthropic format.
        
        Thin wrapper around OpenAIFormatHandler.convert_tools_to_anthropic for backward compatibility.
        
        .. deprecated:: 3.0
            Use self._openai_formatter.convert_tools_to_anthropic() directly instead.
        """
        warnings.warn(
            "_convert_tools_to_anthropic is deprecated and will be removed in v4.0. "
            "Use self._openai_formatter.convert_tools_to_anthropic() directly.",
            DeprecationWarning,
            stacklevel=2
        )
        return self._openai_formatter.convert_tools_to_anthropic(openai_tools)
    
    def _convert_from_anthropic_response(self, anthropic_response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Anthropic response to OpenAI format.
        
        Thin wrapper around AnthropicFormatHandler.convert_from_anthropic_response for backward compatibility.
        
        .. deprecated:: 3.0
            Use self._anthropic_formatter.convert_from_anthropic_response() directly instead.
        """
        warnings.warn(
            "_convert_from_anthropic_response is deprecated and will be removed in v4.0. "
            "Use self._anthropic_formatter.convert_from_anthropic_response() directly.",
            DeprecationWarning,
            stacklevel=2
        )
        return self._anthropic_formatter.convert_from_anthropic_response(anthropic_response)
    
    def _convert_image_to_anthropic(self, image_url: str) -> Optional[str]:
        """Convert image URL to base64 format for Anthropic API.
        
        Fetches image from URL and converts to base64 with size and timeout limits.
        
        Args:
            image_url: URL of the image to convert
            
        Returns:
            Base64-encoded image data URL or None if conversion fails
        """
        MAX_IMAGE_SIZE = 1 * 1024 * 1024  # 1MB limit
        TIMEOUT_SECONDS = 5
        
        try:
            # Validate URL to prevent SSRF attacks
            validated_url = self._openai_formatter.validate_image_url(image_url)
            
            # Create a temporary client for image fetching with timeout
            with httpx.Client(timeout=IMAGE_FETCH_TIMEOUT) as client:
                # Start streaming the image
                with client.stream("GET", validated_url) as response:
                    response.raise_for_status()
                    
                    # Check content-length if available
                    content_length = response.headers.get("content-length")
                    if content_length and int(content_length) > MAX_IMAGE_SIZE:
                        console.print(
                            f"[yellow]Warning: Image too large ({int(content_length):,} bytes). "
                            f"Maximum size is {MAX_IMAGE_SIZE:,} bytes (1MB limit).[/yellow]"
                        )
                        return None
                    
                    # Read image data with size limit
                    chunks = []
                    total_size = 0
                    
                    for chunk in response.iter_bytes(chunk_size=HTTP_CHUNK_SIZE):
                        total_size += len(chunk)
                        if total_size > MAX_IMAGE_SIZE:
                            console.print(
                                f"[yellow]Warning: Image too large (>{MAX_IMAGE_SIZE:,} bytes). "
                                f"Maximum size is 1MB.[/yellow]"
                            )
                            return None
                        chunks.append(chunk)
                    
                    # Combine chunks
                    image_data = b"".join(chunks)
                    
                    # Detect MIME type from content
                    mime_type = self._detect_image_mime_type(image_data)
                    
                    # Convert to base64 data URL
                    base64_data = base64.b64encode(image_data).decode("utf-8")
                    return f"data:{mime_type};base64,{base64_data}"
                    
        except ValidationError as e:
            console.print(
                f"[yellow]Warning: Invalid image URL: {str(e)}[/yellow]"
            )
            return None
        except httpx.TimeoutException:
            console.print(
                f"[yellow]Warning: Image fetch timed out after {TIMEOUT_SECONDS} seconds.[/yellow]"
            )
            return None
        except httpx.HTTPError as e:
            console.print(
                f"[yellow]Warning: Failed to fetch image: {str(e)}[/yellow]"
            )
            return None
        except Exception as e:
            console.print(
                f"[yellow]Warning: Error converting image: {str(e)}[/yellow]"
            )
            return None
    
    def _detect_image_mime_type(self, image_data: bytes) -> str:
        """Detect MIME type from image binary data.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            MIME type string
            
        Raises:
            ValueError: If image format cannot be detected
        """
        if image_data.startswith(b'\xff\xd8\xff'):
            return 'image/jpeg'
        elif image_data.startswith(b'\x89PNG\r\n\x1a\n'):
            return 'image/png'
        elif image_data.startswith(b'GIF8'):
            return 'image/gif'
        elif image_data.startswith(b'RIFF') and b'WEBP' in image_data[:12]:
            return 'image/webp'
        else:
            raise ValueError("Unable to detect image type")
    
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