"""Type definitions for the llm-grok library.

This module contains all TypedDict and type definitions for both OpenAI and Anthropic API formats.
All types follow strict type safety guidelines with no use of Any where avoidable.
"""

from typing import Any, Dict, List, Literal, Optional, Union
from typing_extensions import TypedDict, NotRequired, Required, TypeGuard

# JSON Schema types for better type safety
class JSONSchemaProperty(TypedDict, total=False):
    """JSON Schema property definition."""
    type: str
    description: str
    enum: List[str]
    default: Union[str, int, float, bool, None]
    minimum: Union[int, float]
    maximum: Union[int, float]
    pattern: str
    format: str
    items: "JSONSchemaProperty"
    properties: Dict[str, "JSONSchemaProperty"]
    required: List[str]
    additionalProperties: Union[bool, "JSONSchemaProperty"]


class JSONSchemaParameters(TypedDict):
    """JSON Schema for function parameters."""
    type: Literal["object"]
    properties: Dict[str, JSONSchemaProperty]
    required: NotRequired[List[str]]
    additionalProperties: NotRequired[bool]


class FunctionDefinition(TypedDict):
    """Function definition for tool calling."""
    name: str
    description: str
    parameters: JSONSchemaParameters


class StreamEvent(TypedDict):
    """Structured streaming event."""
    type: Literal["content", "tool_calls", "error", "done"]
    data: NotRequired[Dict[str, Union[str, List["ToolCall"]]]]
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
    # Common types
    "ModelInfo",
    "StreamOptions",
    "ResponseFormat",
    "ToolChoice",
    # Type guards
    "is_model_info_complete",
]


# OpenAI API Types
class ImageContent(TypedDict):
    """Image content in OpenAI message format."""
    type: Literal["image_url"]
    image_url: Dict[str, str]  # {"url": "data:image/jpeg;base64,..." or "https://..."}


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


class ToolCall(TypedDict, total=False):
    """Tool call in OpenAI format."""
    id: str
    type: Literal["function"]
    function: FunctionCallDetails
    index: int


class Message(TypedDict, total=False):
    """OpenAI format message."""
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Union[TextContent, ImageContent]]]
    tool_calls: List[ToolCall]


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
    choices: List[OpenAIChoice]
    usage: OpenAIUsage


# OpenAI Streaming Types
class OpenAIStreamDelta(TypedDict, total=False):
    """Delta content in streaming response."""
    role: Optional[Literal["assistant"]]
    content: Optional[str]
    tool_calls: Optional[List[ToolCall]]


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
    choices: List[OpenAIStreamChoice]


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
    input: Dict[str, Union[str, int, float, bool, List[Any], Dict[str, Any]]]  # Tool-specific parameters


class AnthropicMessage(TypedDict):
    """Anthropic format message."""
    role: Literal["user", "assistant"]
    content: List[Union[AnthropicTextBlock, AnthropicImage, AnthropicToolUse]]


class AnthropicToolChoice(TypedDict):
    """Anthropic tool choice specification."""
    type: Literal["tool"]
    name: str


class AnthropicRequest(TypedDict, total=False):
    """Anthropic API request format."""
    messages: Required[List[AnthropicMessage]]
    system: str
    model: str
    max_tokens: int
    temperature: float
    tools: List["AnthropicToolDefinition"]
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
    content: List[Union[AnthropicTextBlock, AnthropicToolUse]]
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
    function: Dict[Literal["name"], str]


# Request body types
RequestBody = Dict[str, Union[str, int, float, bool, List[Any], Dict[str, Any]]]


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