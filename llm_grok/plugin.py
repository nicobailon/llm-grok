"""Plugin registration for LLM CLI integration."""
from typing import Any, Callable

import llm

from .grok import Grok
from .models import AVAILABLE_MODELS

# Type aliases for clarity
RegisterFunction = Callable[..., None]
CLIObject = Any  # External library, can't be more specific


@llm.hookimpl  # type: ignore[misc]
def register_models(register: RegisterFunction) -> None:
    """Register all available Grok models with LLM CLI.
    
    Args:
        register: The LLM model registration function
    """
    for model_id in AVAILABLE_MODELS:
        register(model_id, Grok)


@llm.hookimpl  # type: ignore[misc]
def register_commands(cli: CLIObject) -> None:
    """Register additional CLI commands.
    
    Currently this plugin doesn't add any custom commands,
    but this hook is required for LLM plugin compatibility.
    
    Args:
        cli: The LLM CLI object
    """
    pass
