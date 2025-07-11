"""llm-grok-enhanced: Enhanced Python plugin for LLM CLI providing advanced access to Grok models.

This package provides enhanced modular components for interacting with xAI's Grok models
with enterprise-grade features, improved reliability, and advanced capabilities.

Original work by Benedikt Hiepler, enhanced and maintained by Nico Bailon.
"""

from rich.console import Console

from .client import GrokClient
from .constants import (
    DEFAULT_RETRY_DELAY,
    MAX_RETRY_DELAY,
)
from .exceptions import (
    APIError,
    AuthenticationError,
    ConversionError,
    GrokError,
    NetworkError,
    QuotaExceededError,
    RateLimitError,
    ValidationError,
)
from .grok import Grok, cleanup_shared_resources
from .models import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    MODEL_INFO,
    get_context_window,
    get_max_output_tokens,
    get_model_capability,
    get_model_info,
    is_tool_capable,
    is_vision_capable,
    validate_model_id,
)
from .plugin import register_commands, register_models
from .types import (
    # Anthropic types
    AnthropicImage,
    AnthropicMessage,
    AnthropicRequest,
    AnthropicToolDefinition,
    FunctionCall,
    FunctionCallDetails,
    # OpenAI types
    ImageContent,
    Message,
    # Common types
    ModelInfo,
    TextContent,
    ToolCall,
    ToolDefinition,
)

# Module-level console instance for output formatting
console = Console()

__version__ = "1.0.0"

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
