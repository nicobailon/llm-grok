"""Integration test to verify the plugin loads correctly."""
import pytest
from typing import cast, Optional, List
import llm
from llm_grok import Grok, register_models, register_commands, GrokError
from llm_grok.types import Message
# Create a local MockPrompt class
class MockPrompt(llm.Prompt):
    """Mock prompt that mimics llm.Prompt interface."""
    def __init__(self, prompt: str, attachments: Optional[List[llm.Attachment]] = None, 
                 system: Optional[str] = None, options: Optional[llm.Options] = None, 
                 model: Optional[llm.Model] = None):
        # Create a minimal model if not provided
        if model is None:
            from llm_grok import Grok
            model = cast(llm.Model, Grok("x-ai/grok-4"))
        
        # Create default options if not provided
        if options is None:
            options = model.Options()
        
        # Initialize parent class
        super().__init__(prompt=prompt, model=model, system=system,  # type: ignore[no-untyped-call]
                         attachments=attachments, options=options)
        
        # Parent class already sets these attributes correctly
        # No need to override them


def test_plugin_registration() -> None:
    """Test that the plugin can be imported and registers correctly."""
    # Test that we can import the main classes
    assert Grok is not None
    assert register_models is not None
    assert register_commands is not None
    assert GrokError is not None


def test_model_instantiation() -> None:
    """Test that we can instantiate a Grok model."""
    model = Grok("x-ai/grok-4")
    assert model.model_id == "x-ai/grok-4"
    assert model.can_stream is True
    assert model.needs_key == "grok"
    assert model.key_env_var == "XAI_API_KEY"


def test_all_models_instantiate() -> None:
    """Test that all available models can be instantiated."""
    from llm_grok import AVAILABLE_MODELS
    
    for model_id in AVAILABLE_MODELS:
        model = Grok(model_id)
        assert model.model_id == model_id


def test_model_capabilities() -> None:
    """Test that model capabilities are accessible."""
    from llm_grok import MODEL_INFO, get_model_capability
    from llm_grok.types import ModelInfo
    
    # Test Grok 4 capabilities
    grok4_info = MODEL_INFO.get("x-ai/grok-4")
    assert grok4_info is not None
    assert grok4_info.get("supports_vision") is True
    assert grok4_info.get("supports_tools") is True
    assert grok4_info.get("context_window") == 256000
    
    # Test using helper function
    assert get_model_capability("x-ai/grok-4", "supports_vision") is True
    assert get_model_capability("x-ai/grok-4", "supports_tools") is True


def test_options_class() -> None:
    """Test that the Options class is properly defined."""
    model = Grok("x-ai/grok-4")
    options = model.Options()
    
    # Check default values
    assert options.temperature == 0.0
    assert options.max_completion_tokens is None
    assert options.tools is None
    assert options.tool_choice is None
    assert options.response_format is None
    assert options.reasoning_effort is None
    assert options.use_messages_endpoint is False


def test_processors_available() -> None:
    """Test that the processors are available and can be instantiated."""
    model = Grok("x-ai/grok-4")
    
    # Check that processors are created
    assert hasattr(model, '_image_processor')
    assert hasattr(model, '_tool_processor')
    assert hasattr(model, '_stream_processor')
    assert hasattr(model, '_openai_formatter')
    assert hasattr(model, '_anthropic_formatter')


def test_build_messages() -> None:
    """Test that build_messages works with basic input."""
    model = Grok("x-ai/grok-4")
    
    # Create a mock prompt using the proper mock class
    prompt = MockPrompt("Hello, world!", model=cast(llm.Model, model))
    messages = model.build_messages(prompt, None)
    
    assert len(messages) == 1
    # Type-safe access - messages[0] is already typed as Message
    user_msg = messages[0]
    assert "role" in user_msg and user_msg["role"] == "user"
    assert "content" in user_msg and user_msg["content"] == "Hello, world!"
    
    # Test with system prompt
    prompt_with_system = MockPrompt("Hello!", system="You are a helpful assistant", model=cast(llm.Model, model))
    messages = model.build_messages(prompt_with_system, None)
    
    assert len(messages) == 2
    # Type-safe access for system message
    system_msg = messages[0]
    assert "role" in system_msg and system_msg["role"] == "system"
    assert "content" in system_msg and system_msg["content"] == "You are a helpful assistant"
    # Type-safe access for user message
    user_msg2 = messages[1]
    assert "role" in user_msg2 and user_msg2["role"] == "user"
    assert "content" in user_msg2 and user_msg2["content"] == "Hello!"


def test_error_classes() -> None:
    """Test that error classes can be raised and caught."""
    from llm_grok import RateLimitError, QuotaExceededError
    
    # Test GrokError
    with pytest.raises(GrokError) as exc_info:
        raise GrokError("Test error")
    assert str(exc_info.value) == "Test error"
    
    # Test RateLimitError
    with pytest.raises(RateLimitError) as exc_info:
        raise RateLimitError("Rate limited", retry_after=60)
    assert "Rate limited" in str(exc_info.value)
    
    # Test QuotaExceededError
    with pytest.raises(QuotaExceededError) as exc_info:
        raise QuotaExceededError("Quota exceeded")
    assert "Quota exceeded" in str(exc_info.value)


if __name__ == "__main__":
    # Run tests
    test_plugin_registration()
    test_model_instantiation()
    test_all_models_instantiate()
    test_model_capabilities()
    test_options_class()
    test_processors_available()
    test_build_messages()
    test_error_classes()
    print("All integration tests passed!")