"""Content processors for LLM Grok.

This module provides the base interfaces and common functionality
for processing various types of content in the Grok plugin.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, TypeVar

# Import base exception from parent for inheritance
from ..exceptions import GrokError

T = TypeVar('T')
R = TypeVar('R')


class ContentProcessor(ABC, Generic[T, R]):
    """Base interface for content processors.
    
    All processors should inherit from this class and implement
    the process method for their specific content type.
    """

    @abstractmethod
    def process(self, content: T) -> R:
        """Process the given content.
        
        Args:
            content: Input content to process
            
        Returns:
            Processed content
            
        Raises:
            ValueError: If content is invalid
            ProcessingError: If processing fails
        """
        pass


class ProcessingError(GrokError):
    """Base exception for processing errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None,
                 error_code: Optional[str] = None):
        super().__init__(message, details=details, error_code=error_code)


class ValidationError(ProcessingError):
    """Raised when content validation fails.
    
    Extends ProcessingError to maintain a consistent error hierarchy
    for processor-specific validation errors.
    """
    def __init__(self, message: str, field: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        # Initialize with validation_error code for consistency
        super().__init__(message, details=details, error_code="validation_error")
        self.field = field


class ProcessorConfig:
    """Configuration for processors."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize processor configuration.
        
        Args:
            **kwargs: Configuration parameters
        """
        self._config = kwargs

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._config[key] = value

    def update(self, **kwargs: Any) -> None:
        """Update multiple configuration values.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self._config.update(kwargs)


class ProcessorRegistry:
    """Registry for managing processors."""

    def __init__(self) -> None:
        """Initialize processor registry."""
        self._processors: Dict[str, ContentProcessor[Any, Any]] = {}

    def register(self, name: str, processor: ContentProcessor[Any, Any]) -> None:
        """Register a processor.
        
        Args:
            name: Processor name
            processor: Processor instance
        """
        self._processors[name] = processor

    def get(self, name: str) -> Optional[ContentProcessor[Any, Any]]:
        """Get a processor by name.
        
        Args:
            name: Processor name
            
        Returns:
            Processor instance or None if not found
        """
        return self._processors.get(name)

    def list(self) -> List[str]:
        """List all registered processor names.
        
        Returns:
            List of processor names
        """
        return list(self._processors.keys())


# Import concrete processors
from .multimodal import ImageProcessor
from .streaming import StreamProcessor
from .tools import ToolProcessor

__all__ = [
    # Base classes and interfaces
    'ContentProcessor',
    'ProcessingError',
    'ValidationError',
    'ProcessorConfig',
    'ProcessorRegistry',
    # Concrete processors
    'ImageProcessor',
    'ToolProcessor',
    'StreamProcessor',
]
