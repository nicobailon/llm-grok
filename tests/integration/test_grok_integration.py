"""Integration tests for Grok model with full pipeline."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Import Stream 1 components
from llm_grok.client import GrokClient
from llm_grok.types import Message, ImageContent, TextContent, ToolCall
from llm_grok.models import is_vision_capable, is_tool_capable
from llm_grok.exceptions import ValidationError, GrokError

# Import Stream 2 components
from llm_grok.processors import (
    ProcessorConfig, ProcessorRegistry,
    ContentProcessor, ProcessingError
)
from llm_grok.processors.multimodal import ImageProcessor
from llm_grok.processors.tools import ToolProcessor
from llm_grok.processors.streaming import StreamProcessor


class TestProcessorIntegration:
    """Test processor integration with Stream 1 components."""
    
    def test_processor_imports_stream1_types(self) -> None:
        """Test that processors can import and use Stream 1 types."""
        # Test type imports
        from llm_grok.types import Message, ToolCall, ImageContent
        
        # Create instances using imported types
        message: Message = {"role": "user", "content": "Hello"}
        image_content: ImageContent = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/image.png"}
        }
        
        assert message["role"] == "user"
        assert image_content["type"] == "image_url"
    
    def test_processor_uses_model_capabilities(self) -> None:
        """Test processors can check model capabilities."""
        # Test vision capability check
        assert is_vision_capable("x-ai/grok-4") is True
        assert is_vision_capable("grok-3-latest") is False
        
        # Test tool capability check
        assert is_tool_capable("x-ai/grok-4") is True
        assert is_tool_capable("grok-2-latest") is False
    
    def test_processor_config_integration(self) -> None:
        """Test ProcessorConfig works with processors."""
        config = ProcessorConfig(
            max_image_size=20 * 1024 * 1024,  # 20MB
            supported_formats=["jpeg", "png", "webp"],
            enable_validation=True
        )
        
        # Create processor with config
        processor = ImageProcessor("x-ai/grok-4", config)
        assert processor.config.get("max_image_size") == 20 * 1024 * 1024
        assert processor.config.get("enable_validation") is True
    
    def test_processor_registry_integration(self) -> None:
        """Test ProcessorRegistry manages processors correctly."""
        registry = ProcessorRegistry()
        
        # Register processors
        image_processor = ImageProcessor("x-ai/grok-4")
        tool_processor = ToolProcessor("x-ai/grok-4")
        stream_processor = StreamProcessor("x-ai/grok-4")
        
        registry.register("image", image_processor)
        registry.register("tool", tool_processor)
        registry.register("stream", stream_processor)
        
        # Verify registration
        assert registry.get("image") is image_processor
        assert registry.get("tool") is tool_processor
        assert registry.get("stream") is stream_processor
        assert set(registry.list()) == {"image", "tool", "stream"}
    
    def test_exception_hierarchy_integration(self) -> None:
        """Test that processor exceptions integrate with Stream 1 exceptions."""
        # ProcessingError should be a GrokError
        error = ProcessingError("Test error")
        assert isinstance(error, GrokError)
        
        # ValidationError should have proper fields
        validation_error = ValidationError(
            "Invalid image format",
            field="image_url",
            details={"format": "bmp", "supported": ["jpeg", "png"]}
        )
        assert validation_error.field == "image_url"
        assert validation_error.details["format"] == "bmp"


class TestClientProcessorIntegration:
    """Test integration between GrokClient and processors."""
    
    @patch('httpx.Client')
    def test_client_with_processor_types(self, mock_httpx_client) -> None:
        """Test GrokClient works with processor-defined types."""
        # Setup mock
        mock_client_instance = MagicMock()
        mock_httpx_client.return_value = mock_client_instance
        
        # Create client
        client = GrokClient("test-api-key")
        
        # Create messages using processor types
        messages: List[Message] = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"}
        ]
        
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test-id",
            "choices": [{
                "message": {"role": "assistant", "content": "Hi there!"},
                "finish_reason": "stop"
            }]
        }
        mock_response.raise_for_status = Mock()
        mock_client_instance.request.return_value = mock_response
        
        # Make request
        result = client.post_openai_completion(
            messages=messages,
            model="x-ai/grok-4",
            stream=False
        )
        
        # When stream=False, result is an httpx.Response
        assert isinstance(result, Mock)  # In tests it's mocked
        assert result.status_code == 200
    
    def test_model_capability_based_processing(self) -> None:
        """Test that processing decisions are based on model capabilities."""
        # Vision-capable model
        if is_vision_capable("x-ai/grok-4"):
            # Would enable image processing
            processor = ImageProcessor("x-ai/grok-4")
            assert processor is not None
        
        # Non-vision model
        if not is_vision_capable("grok-3-latest"):
            # Would skip image processing
            pass
        
        # Tool-capable model
        if is_tool_capable("x-ai/grok-4"):
            # Would enable tool processing
            processor = ToolProcessor("x-ai/grok-4")
            assert processor is not None


class TestFullPipelineIntegration:
    """Test complete pipeline integration scenarios."""
    
    def test_type_flow_through_pipeline(self) -> None:
        """Test that types flow correctly through the pipeline."""
        # Create typed content
        text_content: TextContent = {"type": "text", "text": "Describe this image"}
        image_content: ImageContent = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/cat.jpg"}
        }
        
        # Build message with mixed content
        message: Message = {
            "role": "user",
            "content": [text_content, image_content]
        }
        
        # Verify structure
        assert message["role"] == "user"
        assert isinstance(message["content"], list)
        assert len(message["content"]) == 2
    
    def test_error_propagation(self) -> None:
        """Test that errors propagate correctly through components."""
        # Test validation error
        try:
            raise ValidationError(
                "Invalid image URL",
                field="image_url",
                details={"url": "not-a-url"}
            )
        except GrokError as e:
            # Should be catchable as GrokError
            assert "Invalid image URL" in str(e)
        
        # Test processing error
        try:
            raise ProcessingError("Failed to process content")
        except GrokError as e:
            # Should also be catchable as GrokError
            assert "Failed to process" in str(e)
    


class TestConfigurationIntegration:
    """Test configuration and setup integration."""
    
    def test_processor_configuration_flow(self) -> None:
        """Test configuration flows through processor initialization."""
        # Create shared configuration
        base_config = ProcessorConfig(
            model_id="x-ai/grok-4",
            enable_logging=True,
            strict_validation=True
        )
        
        # Initialize processors with config
        image_proc = ImageProcessor("x-ai/grok-4", base_config)
        tool_proc = ToolProcessor("x-ai/grok-4", base_config)
        stream_proc = StreamProcessor("x-ai/grok-4", base_config)
        
        # Verify config is accessible
        assert image_proc.config.get("model_id") == "x-ai/grok-4"
        assert tool_proc.config.get("enable_logging") is True
        assert stream_proc.config.get("strict_validation") is True
    
    def test_registry_with_configured_processors(self) -> None:
        """Test registry with pre-configured processors."""
        registry = ProcessorRegistry()
        
        # Create configured processors
        config = ProcessorConfig(environment="production")
        
        registry.register("image", ImageProcessor("x-ai/grok-4", config))
        registry.register("tool", ToolProcessor("x-ai/grok-4", config))
        registry.register("stream", StreamProcessor("x-ai/grok-4", config))
        
        # Verify all processors share configuration approach
        image_proc = registry.get("image")
        tool_proc = registry.get("tool")
        stream_proc = registry.get("stream")
        
        # Type assertions to help PyLance understand concrete types
        assert isinstance(image_proc, ImageProcessor)
        assert isinstance(tool_proc, ToolProcessor)
        assert isinstance(stream_proc, StreamProcessor)
        
        # Now PyLance knows these are concrete processor types with config property
        assert image_proc.config.get("environment") == "production"
        assert tool_proc.config.get("environment") == "production"
        assert stream_proc.config.get("environment") == "production"


