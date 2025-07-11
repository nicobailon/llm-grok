"""Main Grok model implementation for LLM CLI integration."""
import sys
import threading
from collections.abc import Iterator
from typing import Any, List, Literal, Optional, Union, cast

import httpx
import llm
from pydantic import Field
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

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
from .models import MODEL_INFO
from .processors import ImageProcessor, StreamProcessor, ToolProcessor
from .types import (
    ImageContent,
    Message,
    RequestBody,
    ResponseFormat,
    TextContent,
    ToolChoice,
    ToolDefinition,
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
                if hasattr(prev_response, 'text') and callable(prev_response.text):
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
        response._prompt_json = {"messages": messages}

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
