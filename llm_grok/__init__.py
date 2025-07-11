"""llm-grok: A Python plugin for LLM CLI providing access to Grok models.

This package provides modular components for interacting with xAI's Grok models
through both OpenAI-compatible and Anthropic-compatible APIs.
"""

from rich.console import Console

# Module-level console instance for output formatting
console = Console()

from .types import (
    # OpenAI types
    ImageContent,
    TextContent,
    FunctionCall,
    FunctionCallDetails,
    ToolCall,
    Message,
    ToolDefinition,
    # Anthropic types
    AnthropicImage,
    AnthropicMessage,
    AnthropicRequest,
    AnthropicToolDefinition,
    # Common types
    ModelInfo,
)

from .models import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    MODEL_INFO,
    get_model_capability,
    validate_model_id,
    get_model_info,
    is_vision_capable,
    is_tool_capable,
    get_context_window,
    get_max_output_tokens,
)

from .exceptions import (
    GrokError,
    RateLimitError,
    QuotaExceededError,
    ValidationError,
    ConversionError,
    APIError,
    AuthenticationError,
    NetworkError,
)

from .constants import (
    DEFAULT_RETRY_DELAY,
    MAX_RETRY_DELAY,
)

from .client import GrokClient
from .grok import Grok, cleanup_shared_resources
from .plugin import register_models, register_commands

__version__ = "2.1.0"

__all__ = [
    # Version
    "__version__",
    # Console
    "console",
    # Core LLM integration
    "Grok",
    "register_models",
    "register_commands",
    # Types
    "ImageContent",
    "TextContent",
    "FunctionCall",
    "FunctionCallDetails",
    "ToolCall",
    "Message",
    "ToolDefinition",
    "AnthropicImage",
    "AnthropicMessage",
    "AnthropicRequest",
    "AnthropicToolDefinition",
    "ModelInfo",
    # Models
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    "MODEL_INFO",
    "get_model_capability",
    "validate_model_id",
    "get_model_info",
    "is_vision_capable",
    "is_tool_capable",
    "get_context_window",
    "get_max_output_tokens",
    # Exceptions
    "GrokError",
    "RateLimitError",
    "QuotaExceededError",
    "ValidationError",
    "ConversionError",
    "APIError",
    "AuthenticationError",
    "NetworkError",
    # Client
    "GrokClient",
    # Cleanup
    "cleanup_shared_resources",
    # Constants
    "DEFAULT_RETRY_DELAY",
    "MAX_RETRY_DELAY",
]