"""Base abstract format handler interface.

This module defines the abstract base class for all format handlers,
establishing the contract for converting between different API formats.
"""

import ipaddress
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from urllib.parse import urlparse, unquote

from ..constants import (
    MIN_BASE64_IMAGE_LENGTH,
    BLOCKED_PORTS,
    AWS_METADATA_ENDPOINT,
)
from ..exceptions import ValidationError
from ..types import (
    AnthropicMessage,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicStreamEvent,
    AnthropicToolDefinition,
    Message,
    OpenAIResponse,
    OpenAIStreamChunk,
    ToolDefinition,
)

__all__ = ["FormatHandler"]


class FormatHandler(ABC):
    """Abstract base class for API format handlers.
    
    Format handlers are responsible for converting between different API formats
    (e.g., OpenAI <-> Anthropic) while maintaining type safety and preserving
    all relevant information.
    """
    
    def __init__(self, model_id: str) -> None:
        """Initialize the format handler.
        
        Args:
            model_id: The model identifier for response metadata
        """
        self.model_id = model_id
    
    @abstractmethod
    def convert_messages_to_anthropic(self, openai_messages: List[Message]) -> AnthropicRequest:
        """Convert OpenAI-style messages to Anthropic format.
        
        This method handles:
        - System message extraction and combination
        - Role mapping (system -> system field)
        - Multimodal content conversion
        - Tool call format conversion
        
        Args:
            openai_messages: List of OpenAI-format messages
            
        Returns:
            Dict with 'messages' and optionally 'system', suitable for Anthropic API
        """
        pass
    
    @abstractmethod
    def convert_tools_to_anthropic(self, openai_tools: List[ToolDefinition]) -> List[AnthropicToolDefinition]:
        """Convert OpenAI tool definitions to Anthropic format.
        
        Args:
            openai_tools: List of OpenAI tool definitions
            
        Returns:
            List of Anthropic tool definitions
        """
        pass
    
    @abstractmethod
    def convert_from_anthropic_response(self, anthropic_response: Dict[str, Any]) -> OpenAIResponse:
        """Convert Anthropic response to OpenAI format.
        
        This method handles:
        - Response structure conversion
        - Content extraction and formatting
        - Tool use to tool calls conversion
        - Usage statistics mapping
        
        Args:
            anthropic_response: Raw Anthropic API response
            
        Returns:
            OpenAI-format response
        """
        pass
    
    @abstractmethod
    def parse_sse_chunk(self, chunk: bytes) -> Iterator[Union[OpenAIStreamChunk, Dict[str, Any]]]:
        """Parse Server-Sent Events chunks from either API format.
        
        This method should handle both OpenAI and Anthropic SSE formats,
        detecting the format and parsing accordingly.
        
        Args:
            chunk: Raw SSE chunk bytes
            
        Yields:
            Parsed event data (format depends on the API)
        """
        pass
    
    @abstractmethod
    def parse_openai_sse(self, buffer: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Parse OpenAI SSE format and return parsed data and remaining buffer.
        
        Args:
            buffer: Current SSE buffer content
            
        Returns:
            Tuple of (parsed_data, remaining_buffer)
        """
        pass
    
    @abstractmethod
    def parse_anthropic_sse(self, buffer: str) -> Tuple[Optional[Tuple[str, Dict[str, Any]]], str]:
        """Parse Anthropic SSE format and return event data and remaining buffer.
        
        Args:
            buffer: Current SSE buffer content
            
        Returns:
            Tuple of ((event_type, event_data), remaining_buffer) or (None, buffer)
        """
        pass
    
    @abstractmethod
    def convert_anthropic_stream_chunk(self, event_type: str, event_data: Dict[str, Any]) -> Optional[OpenAIStreamChunk]:
        """Convert Anthropic streaming event to OpenAI format chunk.
        
        Args:
            event_type: Anthropic event type (e.g., "content_block_delta")
            event_data: Event data payload
            
        Returns:
            OpenAI format chunk or None if event should be skipped
        """
        pass
    
    def validate_image_format(self, image_url: str) -> str:
        """Validate and potentially transform image data format.
        
        This base implementation can be overridden by specific handlers
        if they need custom image validation logic.
        
        Args:
            image_url: Image URL or data URL
            
        Returns:
            Validated/transformed image URL
            
        Raises:
            ValueError: If image format is invalid
        """
        # Basic validation - subclasses can override for more specific logic
        if not image_url:
            raise ValueError("Empty image URL")
        
        if image_url.startswith("data:"):
            # Validate data URL structure
            if "," not in image_url:
                raise ValueError("Invalid data URL format")
            header, _ = image_url.split(",", 1)
            if not header.startswith("data:image/"):
                raise ValueError("Data URL must be an image type")
        elif not image_url.startswith(("http://", "https://")):
            # Assume it's base64 data if not a URL
            if len(image_url) < MIN_BASE64_IMAGE_LENGTH:
                raise ValueError("Invalid image data")
        
        return image_url
    
    def validate_image_url(self, url: str) -> str:
        """Validate URL to prevent SSRF attacks.
        
        This method checks URLs against common SSRF attack patterns including:
        - Localhost and loopback addresses (127.0.0.1, ::1, etc.)
        - Private IP ranges (10.x, 192.168.x, 172.16.x, etc.)
        - Reserved IP addresses
        - Non-HTTP(S) schemes (file://, gopher://, etc.)
        
        Args:
            url: URL to validate
            
        Returns:
            The validated URL (unchanged if valid)
            
        Raises:
            ValidationError: If the URL is potentially dangerous or invalid
        """
        if not url:
            raise ValidationError("Empty URL provided")
        
        # Parse the URL (urlparse automatically handles percent-encoding)
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(f"Invalid URL format: {str(e)}")
        
        # Check scheme - only allow http/https
        if parsed.scheme not in ['http', 'https']:
            raise ValidationError(
                f"Invalid URL scheme '{parsed.scheme}'. Only http/https URLs are allowed."
            )
        
        # Check hostname exists
        if not parsed.hostname:
            raise ValidationError("URL must have a valid hostname")
        
        # Unquote the hostname to handle percent-encoding
        hostname = unquote(parsed.hostname)
        
        # Block localhost and common local aliases
        blocked_hosts = {
            'localhost', '127.0.0.1', '0.0.0.0', '::1', 
            'localhost.localdomain', 'local', AWS_METADATA_ENDPOINT
        }
        if hostname.lower() in blocked_hosts:
            raise ValidationError(
                f"Access to local/internal host '{hostname}' is not allowed"
            )
        
        # Check if hostname is an IP address
        try:
            ip = ipaddress.ip_address(hostname)
            
            # Block private IP ranges (but allow documentation prefix)
            if ip.is_private and not str(ip).startswith('2001:db8'):
                raise ValidationError(
                    f"Access to private IP address '{hostname}' is not allowed"
                )
            
            # Block reserved addresses (but allow documentation prefix)
            if ip.is_reserved and not str(ip).startswith('2001:db8'):
                raise ValidationError(
                    f"Access to reserved IP address '{hostname}' is not allowed"
                )
            
            # Special case for broadcast addresses
            if hasattr(ip, 'is_global') and str(ip) == '255.255.255.255':
                raise ValidationError(
                    f"Access to reserved IP address '{hostname}' is not allowed"
                )
            
            # Block loopback addresses
            if ip.is_loopback:
                raise ValidationError(
                    f"Access to loopback address '{hostname}' is not allowed"
                )
            
            # Block link-local addresses
            if ip.is_link_local:
                raise ValidationError(
                    f"Access to link-local address '{hostname}' is not allowed"
                )
            
            # Block multicast addresses
            if ip.is_multicast:
                raise ValidationError(
                    f"Access to multicast address '{hostname}' is not allowed"
                )
            
        except ValueError:
            # Not an IP address, check for suspicious patterns in hostname
            hostname_lower = hostname.lower()
            
            # Block various forms of localhost
            if any(blocked in hostname_lower for blocked in ['localhost', '127.0.0.1', '::1']):
                raise ValidationError(f"Suspicious hostname pattern detected: {hostname}")
            
            # Block internal-looking domains
            if hostname_lower.endswith(('.local', '.internal', '.localhost', '.lan')):
                raise ValidationError(
                    f"Access to internal domain '{hostname}' is not allowed"
                )
        
        # Check for port restrictions (optional, but recommended for some services)
        if parsed.port:
            # Block commonly exploited internal service ports
            if parsed.port in BLOCKED_PORTS:
                raise ValidationError(
                    f"Access to port {parsed.port} is not allowed"
                )
        
        return url