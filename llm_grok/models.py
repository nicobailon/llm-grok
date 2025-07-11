"""Model registry and configuration for the llm-grok library.

This module contains model definitions, capabilities, and utility functions for working
with different Grok model variants.
"""

from typing import Dict, List, Optional

from .types import ModelInfo

__all__ = [
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
]

# List of all available Grok models
AVAILABLE_MODELS: List[str] = [
    # Grok 4 models
    "x-ai/grok-4",
    "grok-4-heavy",
    # Grok 3 models
    "grok-3-latest",
    "grok-3-fast-latest",
    "grok-3-mini-latest",
    "grok-3-mini-fast-latest",
    # Grok 2 models
    "grok-2-latest",
    "grok-2-vision-latest",
]

# Default model to use if none specified
DEFAULT_MODEL: str = "x-ai/grok-4"

# Model capabilities metadata
MODEL_INFO: Dict[str, ModelInfo] = {
    "x-ai/grok-4": {
        "context_window": 256000,
        "supports_vision": True,
        "supports_tools": True,
        "pricing_tier": "standard",
        "max_output_tokens": 8192,
    },
    "grok-4-heavy": {
        "context_window": 256000,
        "supports_vision": True,
        "supports_tools": True,
        "pricing_tier": "heavy",
        "max_output_tokens": 8192,
    },
    "grok-3-latest": {
        "context_window": 128000,
        "supports_vision": False,
        "supports_tools": False,
        "pricing_tier": "standard",
        "max_output_tokens": 4096,
    },
    "grok-3-fast-latest": {
        "context_window": 128000,
        "supports_vision": False,
        "supports_tools": False,
        "pricing_tier": "standard",
        "max_output_tokens": 4096,
    },
    "grok-3-mini-latest": {
        "context_window": 128000,
        "supports_vision": False,
        "supports_tools": False,
        "pricing_tier": "mini",
        "max_output_tokens": 4096,
    },
    "grok-3-mini-fast-latest": {
        "context_window": 128000,
        "supports_vision": False,
        "supports_tools": False,
        "pricing_tier": "mini",
        "max_output_tokens": 4096,
    },
    "grok-2-latest": {
        "context_window": 32768,
        "supports_vision": False,
        "supports_tools": False,
        "pricing_tier": "standard",
        "max_output_tokens": 4096,
    },
    "grok-2-vision-latest": {
        "context_window": 32768,
        "supports_vision": True,
        "supports_tools": False,
        "pricing_tier": "standard",
        "max_output_tokens": 4096,
    },
}


def get_model_capability(model_id: str, capability: str) -> bool:
    """Get a specific capability for a model.
    
    Args:
        model_id: The model identifier
        capability: The capability to check (e.g., 'supports_vision', 'supports_tools')
        
    Returns:
        True if the model has the capability, False otherwise
    """
    model_info = MODEL_INFO.get(model_id, {})
    return bool(model_info.get(capability, False))


def validate_model_id(model_id: str) -> bool:
    """Validate if a model ID is supported.
    
    Args:
        model_id: The model identifier to validate
        
    Returns:
        True if the model is supported, False otherwise
    """
    return model_id in AVAILABLE_MODELS


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """Get complete model information.
    
    Args:
        model_id: The model identifier
        
    Returns:
        Model information dictionary or None if not found
    """
    return MODEL_INFO.get(model_id)


def is_vision_capable(model_id: str) -> bool:
    """Check if a model supports vision/image inputs.
    
    Args:
        model_id: The model identifier
        
    Returns:
        True if the model supports vision, False otherwise
    """
    return get_model_capability(model_id, "supports_vision")


def is_tool_capable(model_id: str) -> bool:
    """Check if a model supports tool/function calling.
    
    Args:
        model_id: The model identifier
        
    Returns:
        True if the model supports tools, False otherwise
    """
    return get_model_capability(model_id, "supports_tools")


def get_context_window(model_id: str) -> Optional[int]:
    """Get the context window size for a model.
    
    Args:
        model_id: The model identifier
        
    Returns:
        Context window size in tokens or None if not found
    """
    model_info = get_model_info(model_id)
    return model_info.get("context_window") if model_info else None


def get_max_output_tokens(model_id: str) -> Optional[int]:
    """Get the maximum output tokens for a model.
    
    Args:
        model_id: The model identifier
        
    Returns:
        Maximum output tokens or None if not found
    """
    model_info = get_model_info(model_id)
    return model_info.get("max_output_tokens") if model_info else None
