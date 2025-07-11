"""Tests for model registry and utility functions in llm_grok.models module."""

import pytest
from llm_grok.models import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    MODEL_INFO,
    get_model_capability,
    validate_model_id,
    get_model_info,
    is_vision_capable,
    is_tool_capable,
    get_context_window,
    get_max_output_tokens,
)


class TestModelRegistry:
    """Test suite for model registry constants."""
    
    def test_available_models_list(self) -> None:
        """Test AVAILABLE_MODELS contains expected models."""
        assert isinstance(AVAILABLE_MODELS, list)
        assert len(AVAILABLE_MODELS) > 0
        
        # Check for specific model families
        grok4_models = [m for m in AVAILABLE_MODELS if "grok-4" in m]
        grok3_models = [m for m in AVAILABLE_MODELS if "grok-3" in m]
        grok2_models = [m for m in AVAILABLE_MODELS if "grok-2" in m]
        
        assert len(grok4_models) >= 2  # At least grok-4 and grok-4-heavy
        assert len(grok3_models) >= 4  # Multiple grok-3 variants
        assert len(grok2_models) >= 2  # At least grok-2 and grok-2-vision
    
    def test_default_model(self) -> None:
        """Test DEFAULT_MODEL is set correctly."""
        assert DEFAULT_MODEL == "x-ai/grok-4"
        assert DEFAULT_MODEL in AVAILABLE_MODELS
    
    def test_model_info_completeness(self) -> None:
        """Test MODEL_INFO has entries for all available models."""
        for model_id in AVAILABLE_MODELS:
            assert model_id in MODEL_INFO, f"Missing MODEL_INFO for {model_id}"
            
            info = MODEL_INFO[model_id]
            # Required fields
            assert "context_window" in info
            assert "supports_vision" in info
            assert "supports_tools" in info
            
            # Type checks
            assert isinstance(info["context_window"], int)
            assert isinstance(info["supports_vision"], bool)
            assert isinstance(info["supports_tools"], bool)
            
            # Optional fields when present
            if "max_output_tokens" in info:
                assert isinstance(info["max_output_tokens"], int)
            if "pricing_tier" in info:
                assert isinstance(info["pricing_tier"], str)


class TestModelCapabilities:
    """Test suite for model capability checking functions."""
    
    def test_get_model_capability(self) -> None:
        """Test get_model_capability function."""
        # Test Grok 4 capabilities
        assert get_model_capability("x-ai/grok-4", "supports_vision") is True
        assert get_model_capability("x-ai/grok-4", "supports_tools") is True
        
        # Test Grok 3 capabilities
        assert get_model_capability("grok-3-latest", "supports_vision") is False
        assert get_model_capability("grok-3-latest", "supports_tools") is False
        
        # Test non-existent capability
        assert get_model_capability("x-ai/grok-4", "non_existent") is False
        
        # Test non-existent model
        assert get_model_capability("non-existent-model", "supports_vision") is False
    
    def test_validate_model_id(self) -> None:
        """Test validate_model_id function."""
        # Valid models
        assert validate_model_id("x-ai/grok-4") is True
        assert validate_model_id("grok-4-heavy") is True
        assert validate_model_id("grok-3-latest") is True
        assert validate_model_id("grok-2-vision-latest") is True
        
        # Invalid models
        assert validate_model_id("grok-5") is False
        assert validate_model_id("invalid-model") is False
        assert validate_model_id("") is False
    
    def test_get_model_info(self) -> None:
        """Test get_model_info function."""
        # Valid model
        info = get_model_info("x-ai/grok-4")
        assert info is not None
        assert info["context_window"] == 256000
        assert info["supports_vision"] is True
        assert info["supports_tools"] is True
        
        # Invalid model
        assert get_model_info("non-existent") is None


class TestCapabilityHelpers:
    """Test suite for specific capability helper functions."""
    
    def test_is_vision_capable(self) -> None:
        """Test is_vision_capable function."""
        # Vision-capable models
        assert is_vision_capable("x-ai/grok-4") is True
        assert is_vision_capable("grok-4-heavy") is True
        assert is_vision_capable("grok-2-vision-latest") is True
        
        # Non-vision models
        assert is_vision_capable("grok-3-latest") is False
        assert is_vision_capable("grok-2-latest") is False
        
        # Invalid model
        assert is_vision_capable("invalid-model") is False
    
    def test_is_tool_capable(self) -> None:
        """Test is_tool_capable function."""
        # Tool-capable models (only Grok 4 family)
        assert is_tool_capable("x-ai/grok-4") is True
        assert is_tool_capable("grok-4-heavy") is True
        
        # Non-tool models
        assert is_tool_capable("grok-3-latest") is False
        assert is_tool_capable("grok-2-latest") is False
        assert is_tool_capable("grok-2-vision-latest") is False
        
        # Invalid model
        assert is_tool_capable("invalid-model") is False
    
    def test_get_context_window(self) -> None:
        """Test get_context_window function."""
        # Grok 4 models - 256k context
        assert get_context_window("x-ai/grok-4") == 256000
        assert get_context_window("grok-4-heavy") == 256000
        
        # Grok 3 models - 128k context
        assert get_context_window("grok-3-latest") == 128000
        assert get_context_window("grok-3-mini-latest") == 128000
        
        # Grok 2 models - 32k context
        assert get_context_window("grok-2-latest") == 32768
        assert get_context_window("grok-2-vision-latest") == 32768
        
        # Invalid model
        assert get_context_window("invalid-model") is None
    
    def test_get_max_output_tokens(self) -> None:
        """Test get_max_output_tokens function."""
        # Grok 4 models - 8192 max output
        assert get_max_output_tokens("x-ai/grok-4") == 8192
        assert get_max_output_tokens("grok-4-heavy") == 8192
        
        # Grok 3 models - 4096 max output
        assert get_max_output_tokens("grok-3-latest") == 4096
        assert get_max_output_tokens("grok-3-mini-latest") == 4096
        
        # Grok 2 models - 4096 max output
        assert get_max_output_tokens("grok-2-latest") == 4096
        assert get_max_output_tokens("grok-2-vision-latest") == 4096
        
        # Invalid model
        assert get_max_output_tokens("invalid-model") is None


class TestModelConsistency:
    """Test suite for ensuring model data consistency."""
    
    def test_grok4_models_consistency(self) -> None:
        """Test all Grok 4 models have consistent advanced features."""
        grok4_models = ["x-ai/grok-4", "grok-4-heavy"]
        
        for model_id in grok4_models:
            info = get_model_info(model_id)
            assert info is not None
            # All Grok 4 models should support vision and tools
            assert info["supports_vision"] is True
            assert info["supports_tools"] is True
            # All should have 256k context window
            assert info["context_window"] == 256000
            # All should have 8192 max output tokens
            assert info["max_output_tokens"] == 8192
    
    def test_pricing_tiers_present(self) -> None:
        """Test all models have pricing tier information."""
        for model_id in AVAILABLE_MODELS:
            info = get_model_info(model_id)
            assert info is not None
            assert "pricing_tier" in info
            assert info["pricing_tier"] in ["standard", "heavy", "mini"]
    
    def test_vision_models_identified(self) -> None:
        """Test vision-capable models are properly identified."""
        expected_vision_models = [
            "x-ai/grok-4",
            "grok-4-heavy", 
            "grok-2-vision-latest"
        ]
        
        for model_id in expected_vision_models:
            assert is_vision_capable(model_id) is True
            
        # Ensure non-vision models are correctly identified
        non_vision_models = [
            "grok-3-latest",
            "grok-3-fast-latest",
            "grok-2-latest"
        ]
        
        for model_id in non_vision_models:
            assert is_vision_capable(model_id) is False