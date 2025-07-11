"""Multimodal content processing."""
import base64
import logging
from typing import Any, List, Optional, Union

from llm_grok.processors import ContentProcessor, ProcessorConfig, ValidationError

from ..constants import (
    IMAGE_HEADER_BYTES,
    WEBP_HEADER_BYTES,
)
from ..models import is_vision_capable
from ..types import ImageContent, TextContent

logger = logging.getLogger(__name__)


class ImageProcessor(ContentProcessor[Any, List[Union[TextContent, ImageContent]]]):
    """Handles image validation and format conversion.
    
    Supports three image formats:
    1. HTTP/HTTPS URLs - validated and returned as-is
    2. Data URLs with base64 encoding - validated for proper format
    3. Raw base64 strings - MIME type detected and converted to data URLs
    """

    def __init__(self, model_id: str, config: Optional[ProcessorConfig] = None):
        """Initialize image processor."""
        self.model_id = model_id
        self._config = config or ProcessorConfig()

    @property
    def config(self) -> ProcessorConfig:
        """Get processor configuration."""
        return self._config

    def process(self, content: Any) -> List[Union[TextContent, ImageContent]]:
        """Process a prompt with optional image attachments.
        
        Args:
            content: The prompt object containing text and potentially attachments
            
        Returns:
            List of content items formatted for multimodal API requests
            
        Raises:
            ValidationError: If required model is not vision-capable
        """
        # Check if model supports vision
        if not is_vision_capable(self.model_id):
            raise ValidationError(
                f"Model {self.model_id} does not support vision. "
                f"Please use a vision-capable model like grok-4, grok-4-heavy, or grok-2-vision."
            )

        # Extract text and attachments from prompt
        prompt = content
        text = prompt.prompt if hasattr(prompt, 'prompt') else str(prompt)

        # Get attachments from prompt object
        all_attachments = []
        if hasattr(prompt, 'attachments') and prompt.attachments:
            all_attachments.extend(prompt.attachments)

        # If no attachments, return text-only content
        if not all_attachments:
            return [{"type": "text", "text": text}]

        # Build multimodal content
        return self.build_multimodal_content(text, all_attachments)

    def validate(self, content: List[Union[TextContent, ImageContent]]) -> bool:
        """Validate multimodal content structure and formats.
        
        Args:
            content: List of content items to validate
            
        Returns:
            True if all content is valid
            
        Raises:
            ValidationError: If content is invalid
        """
        if not content:
            raise ValidationError("Content list cannot be empty")

        has_text = False
        for item in content:
            if not isinstance(item, dict):
                raise ValidationError(f"Content item must be a dict, got {type(item)}")

            if "type" not in item:
                raise ValidationError("Content item missing 'type' field")

            if item["type"] == "text":
                has_text = True
                if "text" not in item:
                    raise ValidationError("Text content missing 'text' field")
                if not isinstance(item["text"], str):
                    raise ValidationError(f"Text content must be string, got {type(item['text'])}")

            elif item["type"] == "image_url":
                if "image_url" not in item:
                    raise ValidationError("Image content missing 'image_url' field")
                if not isinstance(item["image_url"], dict):
                    raise ValidationError("image_url must be a dict")
                if "url" not in item["image_url"]:
                    raise ValidationError("image_url missing 'url' field")

                # Validate the URL format
                url = item["image_url"]["url"]
                if not isinstance(url, str):
                    raise ValidationError(f"Image URL must be string, got {type(url)}")

                # Basic URL validation
                if not (url.startswith(("http://", "https://", "data:"))):
                    raise ValidationError(
                        "Image URL must be an HTTP(S) URL or data URL"
                    )

            else:
                raise ValidationError(f"Unknown content type: {item['type']}")

        if not has_text:
            raise ValidationError("Content must contain at least one text item")

        return True

    def validate_image_format(self, data: str) -> str:
        """Validate and format image data for API consumption.
        
        Accepts image data in three formats:
        1. HTTP/HTTPS URLs - validated for security and returned
        2. Data URLs with base64 encoding - validated and returned
        3. Raw base64 strings - MIME type detected and formatted as data URL
        
        Args:
            data: The image data as URL, data URL, or base64 string
            
        Returns:
            Properly formatted image URL or data URL
            
        Raises:
            ValidationError: If the image data is invalid or cannot be processed
        """
        if data.startswith(('http://', 'https://')):
            # Validate URL for security using OpenAI format handler
            from ..formats.openai import OpenAIFormatHandler
            handler = OpenAIFormatHandler(self.model_id)
            return handler.validate_image_url(data)
        elif data.startswith('data:'):
            # Data URL - validate format
            if ';base64,' in data:
                return self._validate_data_url(data)
            else:
                raise ValidationError("Invalid data URL format - missing base64 indicator")
        else:
            # Assume raw base64 - try to detect MIME type
            try:
                # Validate and decode base64 once
                decoded = base64.b64decode(data, validate=True)
                mime_type = self._detect_mime_type(decoded)
                return f"data:{mime_type};base64,{data}"
            except Exception as e:
                raise ValidationError(f"Invalid base64 image data: {str(e)}")

    def _validate_data_url(self, data_url: str) -> str:
        """Validate a data URL format.
        
        Args:
            data_url: The data URL to validate
            
        Returns:
            The validated data URL
            
        Raises:
            ValidationError: If the data URL is invalid
        """
        # Basic validation - just ensure it has the right structure
        if not data_url.startswith('data:') or ';base64,' not in data_url:
            raise ValidationError("Invalid data URL format")

        # Extract and validate the base64 portion
        try:
            base64_part = data_url.split(';base64,', 1)[1]
            # Validate it's valid base64
            base64.b64decode(base64_part, validate=True)
        except Exception as e:
            raise ValidationError(f"Invalid base64 in data URL: {str(e)}")

        return data_url

    def _detect_mime_type(self, data: bytes) -> str:
        """Detect MIME type from image data.
        
        Args:
            data: The raw image bytes
            
        Returns:
            The detected MIME type
            
        Raises:
            ValidationError: If the image format is not supported
        """
        # Try to detect image type from magic bytes
        header = data[:IMAGE_HEADER_BYTES]

        if header.startswith(b'\xff\xd8\xff'):
            return 'image/jpeg'
        elif header.startswith(b'\x89PNG'):
            return 'image/png'
        elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
            return 'image/gif'
        elif header.startswith(b'RIFF') and b'WEBP' in data[:WEBP_HEADER_BYTES]:
            return 'image/webp'
        else:
            raise ValidationError(
                "Unable to detect image type from base64 data. "
                "Please provide a data URL with explicit MIME type or use a supported image format "
                "(JPEG, PNG, GIF, or WebP)"
            )

    def build_multimodal_content(self, text: str, attachments: List[Any]) -> List[Union[TextContent, ImageContent]]:
        """Build content array for multimodal messages.
        
        Args:
            text: The text content
            attachments: List of attachments (may include images)
            
        Returns:
            List of content items (text and image_url objects)
        """
        content: List[Union[TextContent, ImageContent]] = [
            {"type": "text", "text": text}
        ]

        for attachment in attachments:
            # Check if this is an image attachment
            if hasattr(attachment, 'type') and attachment.type == "image":
                try:
                    # Get image data from attachment
                    if hasattr(attachment, 'url') and attachment.url:
                        # URL attachment
                        validated_url = self.validate_image_format(attachment.url)
                    elif hasattr(attachment, 'data') and attachment.data:
                        # Direct data attachment
                        validated_url = self.validate_image_format(attachment.data)
                    elif (hasattr(attachment, 'content') or hasattr(attachment, 'path')) and hasattr(attachment, 'base64_content'):
                        # Content or file attachment - convert to base64 data URL
                        base64_data = attachment.base64_content()
                        # Create data URL with image MIME type
                        mime_type = attachment.resolve_type() if hasattr(attachment, 'resolve_type') else "image/jpeg"
                        data_url = f"data:{mime_type};base64,{base64_data}"
                        validated_url = self.validate_image_format(data_url)
                    else:
                        raise ValidationError("Attachment has no content, path, or URL")

                    image_content: ImageContent = {
                        "type": "image_url",
                        "image_url": {"url": validated_url}
                    }
                    content.append(image_content)
                except ValidationError as e:
                    # Log warning but continue with other attachments
                    logger.warning(f"Skipping invalid image: {str(e)}")

        return content

    def process_prompt_with_attachments(self, prompt: Any) -> List[Union[TextContent, ImageContent]]:
        """Process a prompt that may contain image attachments.
        
        Args:
            prompt: The prompt object potentially containing text and attachments
            
        Returns:
            List of content items for multimodal messages
        """
        # Check if prompt has attachments
        if hasattr(prompt, 'attachments') and prompt.attachments:
            return self.build_multimodal_content(prompt.prompt, prompt.attachments)

        # Return text-only content
        return [{"type": "text", "text": prompt.prompt}]
