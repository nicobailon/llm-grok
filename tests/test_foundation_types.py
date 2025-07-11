"""Foundation type tests for the llm-grok library.

This module tests the fundamental type definitions and ensures they work correctly
with proper type safety and runtime validation.
"""

from typing import get_type_hints

from llm_grok.types import (
    AnthropicContent,
    AnthropicResponse,
    EnhancedModelInfo,
    HTTPClient,
    HTTPResponse,
    LLMModelProtocol,
    LLMOptionsProtocol,
    LLMPromptProtocol,
    ModelInfo,
    OpenAIResponse,
    RequestBody,
    is_model_info_complete,
)


def test_openai_response_structure() -> None:
    """Verify OpenAI response types are properly defined."""
    hints = get_type_hints(OpenAIResponse)
    assert "choices" in hints
    assert "id" in hints
    assert "model" in hints
    assert "created" in hints
    assert "object" in hints
    assert "usage" in hints


def test_anthropic_response_structure() -> None:
    """Verify Anthropic response types are properly defined."""
    hints = get_type_hints(AnthropicResponse)
    assert "content" in hints
    assert "id" in hints
    assert "type" in hints
    assert "role" in hints
    assert "model" in hints
    assert "usage" in hints


def test_request_body_structure() -> None:
    """Verify request body supports both API formats."""
    hints = get_type_hints(RequestBody)
    assert "model" in hints
    assert "messages" in hints
    # Optional fields should still be in the type hints
    assert "tools" in hints
    assert "temperature" in hints
    assert "max_completion_tokens" in hints
    assert "max_tokens" in hints  # Anthropic compatibility


def test_model_info_completeness() -> None:
    """Verify ModelInfo has all required capabilities."""
    hints = get_type_hints(ModelInfo)
    required_fields = {
        "context_window", "max_output_tokens", 
        "supports_tools", "supports_vision", "pricing_tier"
    }
    assert required_fields.issubset(set(hints.keys()))


def test_enhanced_model_info_completeness() -> None:
    """Verify EnhancedModelInfo has additional capabilities."""
    hints = get_type_hints(EnhancedModelInfo)
    required_fields = {
        "context_window", "max_output_tokens", 
        "supports_tools", "supports_vision", "supports_streaming",
        "pricing_tier"
    }
    assert required_fields.issubset(set(hints.keys()))
    
    # Check for additional fields
    assert "supports_reasoning" in hints
    assert "api_format" in hints


def test_anthropic_content_structure() -> None:
    """Verify Anthropic content block structure."""
    hints = get_type_hints(AnthropicContent)
    assert "type" in hints
    # Optional fields
    assert "text" in hints
    assert "id" in hints
    assert "name" in hints
    assert "input" in hints


def test_protocol_interfaces() -> None:
    """Verify protocol interfaces are properly defined."""
    # LLM Model Protocol
    llm_model_hints = get_type_hints(LLMModelProtocol)
    assert "model_id" in llm_model_hints
    
    # LLM Options Protocol
    llm_options_hints = get_type_hints(LLMOptionsProtocol)
    assert "temperature" in llm_options_hints
    assert "max_tokens" in llm_options_hints
    
    # LLM Prompt Protocol
    llm_prompt_hints = get_type_hints(LLMPromptProtocol)
    assert "prompt" in llm_prompt_hints
    assert "attachments" in llm_prompt_hints


def test_http_protocol_interfaces() -> None:
    """Verify HTTP protocol interfaces are properly defined."""
    # HTTP Response Protocol
    response_hints = get_type_hints(HTTPResponse)
    assert "status_code" in response_hints
    assert "headers" in response_hints
    
    # HTTP Client Protocol
    client_hints = get_type_hints(HTTPClient)
    # Methods are not included in get_type_hints for protocols
    # but we can verify the protocol exists
    assert hasattr(HTTPClient, "__protocol__")


def test_model_info_type_guard() -> None:
    """Test the ModelInfo type guard function."""
    # Valid ModelInfo
    valid_model_info = {
        "context_window": 8192,
        "supports_vision": True,
        "supports_tools": False,
        "pricing_tier": "standard",
        "max_output_tokens": 4096
    }
    assert is_model_info_complete(valid_model_info)
    
    # Invalid - missing field
    invalid_model_info = {
        "context_window": 8192,
        "supports_vision": True,
        # Missing other required fields
    }
    assert not is_model_info_complete(invalid_model_info)
    
    # Invalid - wrong type
    invalid_type = {
        "context_window": "8192",  # Should be int
        "supports_vision": True,
        "supports_tools": False,
        "pricing_tier": "standard",
        "max_output_tokens": 4096
    }
    assert not is_model_info_complete(invalid_type)
    
    # Not a dict
    assert not is_model_info_complete("not a dict")
    assert not is_model_info_complete(None)


def test_type_definitions_have_docstrings() -> None:
    """Verify important type definitions have docstrings."""
    assert OpenAIResponse.__doc__ is not None
    assert AnthropicResponse.__doc__ is not None
    assert RequestBody.__doc__ is not None
    assert EnhancedModelInfo.__doc__ is not None
    assert LLMModelProtocol.__doc__ is not None


def test_runtime_protocol_checking() -> None:
    """Test runtime protocol checking capabilities."""
    # Test that we can check protocol compliance at runtime
    from typing import runtime_checkable
    
    # These should be runtime checkable
    assert hasattr(LLMModelProtocol, "__instancecheck__")
    assert hasattr(LLMOptionsProtocol, "__instancecheck__")
    assert hasattr(LLMPromptProtocol, "__instancecheck__")
    
    # Create mock objects to test protocol compliance
    class MockModel:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
        
        def execute(self, prompt: object, options: object) -> str:
            return "mock response"
    
    mock_model = MockModel("test-model")
    assert isinstance(mock_model, LLMModelProtocol)
    
    class MockOptions:
        def __init__(self) -> None:
            self.temperature = 0.5
            self.max_tokens = 1000
    
    mock_options = MockOptions()
    assert isinstance(mock_options, LLMOptionsProtocol)