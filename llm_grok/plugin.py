"""Plugin registration for LLM CLI integration."""
from typing import Any

import llm

from .grok import Grok
from .models import AVAILABLE_MODELS


@llm.hookimpl
def register_models(register: Any) -> None:
    """Register all available Grok models with LLM CLI.
    
    Args:
        register: The LLM model registration function
    """
    for model_id in AVAILABLE_MODELS:
        register(Grok(model_id), aliases=(model_id,))


@llm.hookimpl
def register_commands(cli: Any) -> None:
    """Register additional CLI commands.
    
    Currently this plugin doesn't add any custom commands,
    but this hook is required for LLM plugin compatibility.
    
    Args:
        cli: The LLM CLI object
    """
    pass
