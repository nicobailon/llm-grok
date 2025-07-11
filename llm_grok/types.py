"""Type definitions for the llm-grok library.

This module contains all TypedDict and type definitions for both OpenAI and Anthropic API formats.
All types follow strict type safety guidelines with no use of Any where avoidable.
"""

from typing import Any, Literal, Optional, Union, Protocol, runtime_checkable
from collections.abc import Iterator
from contextlib import AbstractContextManager

from typing_extensions import NotRequired, Required, TypedDict, TypeGuard


# JSON Schema types for better type safety
class JSONSchemaProperty(TypedDict, total=False):
    """JSON Schema property definition."""
    type: str
    description: str
    enum: list[str]
    default: Union[str, int, float, bool, None]
    minimum: Union[int, float]
    maximum: Union[int, float]
    pattern: str
    format: str
    items: "JSONSchemaProperty"
    properties: dict[str, "JSONSchemaProperty"]
    required: list[str]
    additionalProperties: Union[bool, "JSONSchemaProperty"]


class JSONSchemaParameters(TypedDict):
    """JSON Schema for function parameters."""
    type: Literal["object"]
    properties: dict[str, JSONSchemaProperty]
    required: NotRequired[list[str]]
    additionalProperties: NotRequired[bool]


class FunctionDefinition(TypedDict):
    """Function definition for tool calling."""
    name: str
    description: str
    parameters: JSONSchemaParameters


class StreamEvent(TypedDict):
    """Structured streaming event."""
    type: Literal["content", "tool_calls", "error", "done"]
    data: NotRequired[dict[str, Union[str, list["ToolCall"]]]]
    error: NotRequired[str]


__all__ = [
    # JSON Schema types
    "JSONSchemaProperty",
    "JSONSchemaParameters",
    "FunctionDefinition",
    "StreamEvent",
    # OpenAI types
    "ImageContent",
    "TextContent",
    "FunctionCall",
    "FunctionCallDetails",
    "ToolCall",
    "ToolCallWithIndex", 
    "RawFunctionCall",
    "RawToolCall", 
    "BaseMessage",
    "Message",
    "ToolDefinition",
    "OpenAIResponse",
    "OpenAIStreamChoice",
    "OpenAIStreamDelta",
    "OpenAIChoice",
    "OpenAIUsage",
    "OpenAIStreamChunk",
    # Anthropic types
    "AnthropicImageSource",
    "AnthropicImage",
    "AnthropicTextBlock",
    "AnthropicToolUse",
    "AnthropicMessage",
    "AnthropicToolChoice",
    "AnthropicRequest",
    "AnthropicToolDefinition",
    "AnthropicResponse",
    "AnthropicStreamDelta",
    "AnthropicStreamEvent",
    "AnthropicUsage",
    "AnthropicContent",
    # Common types
    "ModelInfo",
    "EnhancedModelInfo",
    "StreamOptions",
    "ResponseFormat",
    "ToolChoice",
    "FunctionChoice",
    "RequestBody",
    # Protocol interfaces
    "LLMModelProtocol",
    "LLMOptionsProtocol", 
    "LLMPromptProtocol",
    "HTTPResponse",
    "HTTPClient",
    # Type guards
    "is_model_info_complete",
]


# OpenAI API Types
class ImageContent(TypedDict):
    """Image content in OpenAI message format."""
    type: Literal["image_url"]
    image_url: dict[str, str]  # {"url": "data:image/jpeg;base64,..." or "https://..."}


class TextContent(TypedDict):
    """Text content in OpenAI message format."""
    type: Literal["text"]
    text: str


class FunctionCallDetails(TypedDict):
    """Details of a function call."""
    name: str
    arguments: str  # JSON string of function arguments


class FunctionCall(TypedDict, total=False):
    """Function call in message content."""
    id: str
    type: Literal["function"]
    function: FunctionCallDetails


class ToolCall(TypedDict):
    """Tool call in OpenAI format - required fields only."""
    id: str
    type: Literal["function"]
    function: FunctionCallDetails


class ToolCallWithIndex(ToolCall, total=False):
    """Tool call with optional index for streaming accumulation."""
    index: int


# Raw API response types for proper typing during processing
class RawFunctionCall(TypedDict, total=False):
    """Raw function call data from API responses."""
    name: str
    arguments: str


class RawToolCall(TypedDict, total=False):
    """Raw tool call data from API responses."""
    id: str
    type: str
    function: RawFunctionCall
    index: int  # For streaming responses


class BaseMessage(TypedDict):
    """Core message fields that are always required."""
    role: Literal["system", "user", "assistant"]
    content: Union[str, list[Union[TextContent, ImageContent]]]


class Message(BaseMessage, total=False):
    """Complete message with optional fields."""
    tool_calls: list[ToolCall]


class ToolDefinition(TypedDict):
    """OpenAI tool definition format."""
    type: Literal["function"]
    function: FunctionDefinition


# OpenAI Response Types
class OpenAIChoice(TypedDict):
    """Choice in OpenAI response."""
    index: int
    message: Message
    finish_reason: Optional[str]


class OpenAIUsage(TypedDict):
    """Usage information in OpenAI response."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIResponse(TypedDict):
    """Complete OpenAI API response."""
    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage


# OpenAI Streaming Types
class OpenAIStreamDelta(TypedDict, total=False):
    """Delta content in streaming response."""
    role: Optional[Literal["assistant"]]
    content: Optional[str]
    tool_calls: Optional[list[ToolCall]]


class OpenAIStreamChoice(TypedDict):
    """Choice in OpenAI streaming response."""
    index: int
    delta: OpenAIStreamDelta
    finish_reason: Optional[str]


class OpenAIStreamChunk(TypedDict):
    """Chunk in OpenAI streaming response."""
    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIStreamChoice]


# Anthropic API Types
class AnthropicImageSource(TypedDict):
    """Image source in Anthropic format."""
    type: Literal["base64"]
    media_type: str  # e.g., "image/jpeg", "image/png"
    data: str  # base64 encoded image data


class AnthropicImage(TypedDict):
    """Image content in Anthropic message format."""
    type: Literal["image"]
    source: AnthropicImageSource


class AnthropicTextBlock(TypedDict):
    """Text block in Anthropic message format."""
    type: Literal["text"]
    text: str


class AnthropicToolUse(TypedDict):
    """Tool use block in Anthropic format."""
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Union[str, int, float, bool, list[Any], dict[str, Any]]]  # Tool-specific parameters


class AnthropicMessage(TypedDict):
    """Anthropic format message."""
    role: Literal["user", "assistant"]
    content: list[Union[AnthropicTextBlock, AnthropicImage, AnthropicToolUse]]


class AnthropicToolChoice(TypedDict):
    """Anthropic tool choice specification."""
    type: Literal["tool"]
    name: str


class AnthropicRequest(TypedDict, total=False):
    """Anthropic API request format."""
    messages: Required[list[AnthropicMessage]]
    system: str
    model: str
    max_tokens: int
    temperature: float
    tools: list["AnthropicToolDefinition"]
    tool_choice: Union[Literal["auto", "none"], AnthropicToolChoice]


class AnthropicToolDefinition(TypedDict):
    """Anthropic tool definition format."""
    name: str
    description: str
    input_schema: JSONSchemaParameters  # JSON Schema for tool parameters


# Anthropic Response Types
class AnthropicUsage(TypedDict):
    """Usage information in Anthropic response."""
    input_tokens: int
    output_tokens: int


class AnthropicResponse(TypedDict):
    """Complete Anthropic API response."""
    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    content: list[Union[AnthropicTextBlock, AnthropicToolUse]]
    model: str
    stop_reason: Optional[str]
    stop_sequence: Optional[str]
    usage: AnthropicUsage


# Anthropic Streaming Types
class AnthropicStreamDelta(TypedDict, total=False):
    """Delta content in Anthropic streaming."""
    text: str
    partial_json: str


class AnthropicStreamEvent(TypedDict):
    """Event in Anthropic streaming response."""
    type: str  # e.g., "message_start", "content_block_start", "content_block_delta"
    index: Optional[int]
    message: Optional[AnthropicResponse]
    content_block: Optional[Union[AnthropicTextBlock, AnthropicToolUse]]
    delta: Optional[AnthropicStreamDelta]
    usage: Optional[AnthropicUsage]


# Model Configuration Types
class ModelInfo(TypedDict):
    """Information about a specific model."""
    context_window: int
    supports_vision: bool
    supports_tools: bool
    max_output_tokens: int
    pricing_tier: str


# Stream Processing Types
class StreamOptions(TypedDict, total=False):
    """Options for stream processing."""
    include_usage: bool


# Common Request Types
class ResponseFormat(TypedDict):
    """Response format specification."""
    type: Literal["json_object", "text"]


class ToolChoice(TypedDict):
    """Tool choice specification for OpenAI format."""
    type: Literal["function"]
    function: dict[Literal["name"], str]




# Protocol interfaces for external libraries
@runtime_checkable
class LLMModelProtocol(Protocol):
    """Protocol for LLM framework model interface."""
    model_id: str
    
    def execute(self, prompt: Any, options: Any) -> Any: ...
    def __init__(self, model_id: str) -> None: ...


@runtime_checkable  
class LLMOptionsProtocol(Protocol):
    """Protocol for LLM framework options."""
    temperature: float | None
    max_tokens: int | None


@runtime_checkable
class LLMPromptProtocol(Protocol):
    """Protocol for LLM framework prompt."""
    prompt: str | list[Any]
    attachments: list[Any] | None


class HTTPResponse(Protocol):
    """Protocol for HTTP response objects."""
    status_code: int
    headers: dict[str, str]
    
    def raise_for_status(self) -> None: ...
    def json(self) -> dict[str, Any]: ...
    def iter_lines(self) -> Iterator[str]: ...


class HTTPClient(Protocol):
    """Protocol for HTTP client objects."""
    
    def post(
        self, 
        url: str, 
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None
    ) -> HTTPResponse: ...
    
    def stream(
        self,
        method: str,
        url: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None
    ) -> AbstractContextManager[HTTPResponse]: ...


# Enhanced unified request body type
class RequestBody(TypedDict):
    """Unified request body for both APIs."""
    model: str
    messages: list[Message]
    temperature: NotRequired[float]
    max_completion_tokens: NotRequired[int]
    max_tokens: NotRequired[int]  # For Anthropic compatibility
    tools: NotRequired[list[ToolDefinition]]
    tool_choice: NotRequired[Literal["auto", "none"] | ToolChoice]
    response_format: NotRequired[ResponseFormat]
    reasoning_effort: NotRequired[str]
    stream: NotRequired[bool]


# Enhanced model info with additional capabilities
class EnhancedModelInfo(TypedDict):
    """Enhanced model information with all capabilities."""
    context_window: int
    max_output_tokens: int
    supports_tools: bool
    supports_vision: bool
    supports_streaming: bool
    supports_reasoning: NotRequired[bool]
    api_format: NotRequired[Literal["openai", "anthropic"]]
    pricing_tier: str


# Content types for Anthropic API compatibility
class AnthropicContent(TypedDict):
    """Anthropic content block structure."""
    type: Literal["text", "tool_use"]
    text: NotRequired[str]
    id: NotRequired[str]
    name: NotRequired[str]
    input: NotRequired[dict[str, Any]]


# Enhanced Tool Choice for specific function selection
class FunctionChoice(TypedDict):
    """Function choice specification."""
    name: str


# Type guards for runtime safety
def is_model_info_complete(obj: Any) -> TypeGuard[ModelInfo]:
    """Type guard to check if an object is a complete ModelInfo.
    
    Args:
        obj: Object to check
        
    Returns:
        True if the object has all required ModelInfo fields
    """
    if not isinstance(obj, dict):
        return False

    required_keys = {"context_window", "supports_vision", "supports_tools", "pricing_tier", "max_output_tokens"}
    if not all(key in obj for key in required_keys):
        return False

    # Type check each field
    return (
        isinstance(obj.get("context_window"), int) and
        isinstance(obj.get("supports_vision"), bool) and
        isinstance(obj.get("supports_tools"), bool) and
        isinstance(obj.get("pricing_tier"), str) and
        isinstance(obj.get("max_output_tokens"), int)
    )
