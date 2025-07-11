# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `llm-grok`, a Python plugin for the [LLM](https://llm.datasette.io/) CLI tool that provides access to Grok models using the xAI API. The plugin is published to PyPI and enables users to interact with various Grok models through the command line.

**Current Version**: 2.1.0 (with /messages endpoint support)

**Key Features**:
- **Grok 4 Support**: Full support for xAI's latest Grok 4 models with 256k context window
- **Multimodal Capabilities**: Image analysis support for vision-enabled models
- **Function Calling**: Tool/function calling support with parallel execution
- **Advanced Reasoning**: Support for Grok 4's enhanced reasoning capabilities
- **Dual Endpoint Support**: Both OpenAI-compatible and Anthropic-compatible /messages endpoints
- **Backwards Compatibility**: Maintains support for all Grok 2 and Grok 3 models
- **Rich Terminal UI**: Enhanced user experience with progress indicators and error handling

## Development Commands

### Setup and Dependencies

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -e '.[test]'
```

### Testing

```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_grok.py

# Run specific test
pytest tests/test_grok.py::test_model_initialization
```

### Type Checking

```bash
# Run type checker
mypy llm_grok --strict
```

### Building and Publishing

```bash
# Build the package
python -m build

# Upload to PyPI (requires credentials)
python -m twine upload dist/*
```

## Architecture

The llm-grok plugin uses a modular architecture optimized for maintainability and enterprise use:

### **Core Structure**
```
llm_grok/
├── grok.py              # Main model class and orchestration
├── client.py            # Enterprise HTTP client  
├── models.py            # Model registry and capabilities
├── types.py             # Comprehensive type definitions
├── exceptions.py        # Error handling hierarchy
├── formats/             # API format converters
│   ├── openai.py       # OpenAI format handling
│   └── anthropic.py    # Anthropic format handling
└── processors/          # Content processors
    ├── multimodal.py   # Image processing
    ├── streaming.py    # SSE stream handling
    └── tools.py        # Function calling
```

### **Development Guidelines**

**Always use the modular implementation** for all development work:
- Modify components in their specialized files
- Add new processors for new content types  
- Extend formatters for new API formats
- Use the type system for safety


### **Key Patterns**

1. **Processor Pattern**: Specialized classes handle specific content types
2. **Formatter Pattern**: Convert between different API formats
3. **Type Safety**: Extensive TypedDict usage throughout
4. **Enterprise Reliability**: Circuit breakers, retries, connection pooling

### Model Configuration

- Available models include all Grok 2, 3, and 4 variants
- Default model: `x-ai/grok-4` 
- Model metadata includes:
  - Context window sizes (up to 256k tokens for Grok 4)
  - Vision support capabilities
  - Function calling support flags
- Supports streaming and non-streaming responses

### Enhanced Options

- `temperature` (0-1): Sampling temperature control
- `max_completion_tokens`: Maximum tokens to generate
- `tools`: Function/tool definitions for function calling
- `tool_choice`: Control over function selection ("auto", "none", or specific function)
- `response_format`: Structured output format (e.g., JSON mode)
- `reasoning_effort`: Control over reasoning depth
- `use_messages_endpoint`: Enable Anthropic-compatible /messages endpoint

### API Integration

- Primary URL: `https://api.x.ai/v1/chat/completions` (OpenAI-compatible)
- Alternative URL: `https://api.x.ai/v1/messages` (Anthropic-compatible)
- Uses OpenAI-compatible chat completions API by default
- Authentication via Bearer token from `XAI_API_KEY` or stored key
- Multimodal message format for image inputs
- Function calling with parallel tool execution
- Full format conversion between OpenAI and Anthropic APIs

### Key Design Patterns

1. **Rate Limit Handling**: Automatic retry with exponential backoff and visual progress indicators using Rich
2. **Error Handling**: Graceful error messages with helpful user guidance
3. **Streaming Support**: Parses SSE (Server-Sent Events) format for real-time responses
4. **Message Building**: Constructs proper message arrays including system prompts and conversation history
5. **Model Capability Detection**: Uses `MODEL_INFO` registry to enable/disable features per model
6. **Multimodal Content Handling**: Validates and formats images for vision-capable models only
7. **Function Call Management**: Handles both streaming and non-streaming tool calls with proper accumulation
8. **Image Validation**: Robust validation for URLs, base64 data, and data URLs with MIME type detection
9. **Endpoint Flexibility**: Seamless switching between OpenAI and Anthropic endpoints with format conversion
10. **Stream Parsing**: Separate helper functions for parsing OpenAI and Anthropic SSE formats

## Grok 4 Features

### Multimodal Support

The plugin now supports image analysis for vision-capable models (Grok 4, Grok 4 Heavy, and Grok 2 Vision):

- **Image Formats**: Supports URLs, base64 encoded images, and data URLs
- **MIME Type Detection**: Automatically detects and formats JPEG, PNG, GIF, and WebP images
- **Model Filtering**: Only enables multimodal features for models with `supports_vision: True`
- **Error Handling**: Graceful handling of invalid images with user-friendly warnings

### Function Calling

Full support for OpenAI-compatible function calling:

- **Tool Definitions**: JSON Schema-based function definitions
- **Parallel Execution**: Support for multiple simultaneous function calls
- **Streaming Support**: Real-time tool call accumulation in streaming mode
- **Tool Choice Control**: Options for automatic, manual, or disabled tool selection
- **Model Filtering**: Only available for models with `supports_tools: True`

### Enhanced Context Window

- **256k Token Context**: Grok 4 models support up to 256,000 token context windows
- **Automatic Handling**: No special configuration needed for large contexts
- **Backwards Compatibility**: Older models maintain their original context limits

### Reasoning Capabilities

- **Reasoning Effort Control**: Optional `reasoning_effort` parameter for controlling reasoning depth
- **Automatic Reasoning**: Grok 4 models automatically engage reasoning when beneficial
- **Transparent Operation**: Reasoning tokens are handled transparently by the API

## Testing Strategy

Tests use `pytest` with `pytest-httpx` for mocking HTTP requests. All API calls are mocked to avoid requiring real API keys during testing. The expanded test suite covers:

**Core Functionality**:
- Model initialization and configuration for all model variants
- Message building with/without system prompts and conversations
- Streaming and non-streaming requests
- Options handling (temperature, max_completion_tokens)
- Error scenarios and API error parsing

**Grok 4 Features**:
- Model metadata registry validation
- Grok 4 model initialization and capabilities
- Default model verification (x-ai/grok-4)

**Multimodal Support**:
- Image URL handling and validation
- Base64 image processing with MIME type detection
- Data URL format validation
- Model capability filtering (vision vs non-vision models)
- Error handling for invalid image formats

**Function Calling**:
- Tool definition validation
- Function calling request formatting
- Tool response handling in streaming and non-streaming modes
- Model capability filtering (tools vs non-tools models)
- Parallel tool call accumulation

**Test Coverage**: 20+ test functions covering all major features with comprehensive edge case handling.

## Development Patterns and Examples

### Model Capability Checking

Always check model capabilities before enabling features:

```python
model_info = MODEL_INFO.get(self.model_id, {})
supports_vision = model_info.get("supports_vision", False)
supports_tools = model_info.get("supports_tools", False)

if supports_vision and hasattr(prompt, 'attachments'):
    # Enable multimodal processing

if supports_tools and options.tools:
    # Include function calling parameters
```

### Multimodal Message Building

For vision-capable models, messages with images use array format:

```python
# Text-only message (all models)
{"role": "user", "content": "Hello"}

# Multimodal message (vision models only)
{
    "role": "user",
    "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
    ]
}
```

### Image Validation Pattern

Robust image handling with validation:

```python
try:
    formatted_url = self._validate_image_format(attachment.data)
    content.append({
        "type": "image_url",
        "image_url": {"url": formatted_url}
    })
except ValueError as e:
    # Log warning but continue processing
    console.print(f"[yellow]Warning: Skipping invalid image - {str(e)}[/yellow]")
```

### Function Calling Request Format

For tool-capable models, include function definitions:

```python
body = {
    "model": "x-ai/grok-4",
    "messages": messages,
    "tools": [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }],
    "tool_choice": "auto"
}
```

### Tool Response Handling

Handle both streaming and non-streaming tool calls:

```python
# Non-streaming
if "tool_calls" in message and message["tool_calls"]:
    response.tool_calls = message["tool_calls"]

# Streaming - accumulate tool calls
if "tool_calls" in delta:
    for tool_call in delta["tool_calls"]:
        # Merge streaming tool call data
        index = tool_call.get("index", 0)
        # ... accumulation logic
```

## Important Notes

- The plugin requires an xAI API key, obtained from x.ai
- Supports both environment variable (`XAI_API_KEY`) and LLM key storage
- Implements proper rate limiting to handle API quotas gracefully
- Uses Rich for enhanced terminal output and progress indicators
- **Default model changed**: Now defaults to `x-ai/grok-4` (breaking change from v1.0)
- **Multimodal support**: Image analysis available for Grok 4, Grok 4 Heavy, and Grok 2 Vision
- **Function calling**: Available for Grok 4 models with full OpenAI compatibility
- **Large context**: Grok 4 models support up to 256k token context windows
- **Backwards compatibility**: All existing Grok 2 and Grok 3 models remain supported

## Implementation Choice

When working on this codebase:
- **Use the modular implementation** (`llm_grok/` package) for all development work
- Focus on the production modular architecture for all development work
- Use the comprehensive documentation in ARCHITECTURE.md to understand core concepts



# Python Type Safety & Development Guidelines

## Type Safety Rules

**FORBIDDEN Anti-Patterns:**
```python
# ❌ NEVER: Type widening, Any, type: ignore, removing constraints
def process(data: Any) -> Any: ...  # NO
user: Any = get_user()  # NO
result = operation()  # type: ignore  # NO
Status = str  # NO - use Literal/Enum
```

**REQUIRED Practices:**
```python
# ✅ ALWAYS: Precise types
from typing import Literal, NewType, TypedDict, TypeGuard, Protocol
from enum import Enum

class UserRole(Enum):
    ADMIN = "admin"
    USER = "user"

class UserData(TypedDict):
    email: str
    role: UserRole
    status: Literal["active", "inactive"]

UserId = NewType('UserId', str)
Email = NewType('Email', str)

def is_user_data(obj: object) -> TypeGuard[UserData]:
    return isinstance(obj, dict) and "email" in obj
```

## Resolution Protocol

1. **Analyze**: Missing annotations, generic constraints, Union guards, Protocols, Optional handling
2. **Fix (in order)**: Explicit annotations → TypeGuards → Generic constraints → Protocols → Literals → TypedDict/dataclasses
3. **Validate**: `mypy --strict` zero errors, PyLance clean, business constraints in types
4. **Last resort**: Document failures, try TypeVar bounds, Protocols, overloads, guards

## TDD Protocol

### Phase 1: Test First
```python
# Write tests including type validation
def test_function_signature():
    hints = get_type_hints(my_function)
    assert hints['return'] == UserData
    
def test_type_safety():
    with pytest.raises(TypeError):
        process_user("invalid")
```
**State**: "Doing TDD - do NOT create mock implementations"

### Phase 2-4: Validate → Implement → Verify
- Run tests (must fail) → Write code → All tests/types pass → Runtime type verification

## Pydantic/Dataclass Standards

```python
# ✅ Precise Pydantic
class User(BaseModel):
    id: UserId
    email: Email = Field(..., regex=r'^[^@]+@[^@]+\.[^@]+$')
    role: UserRole
    status: Literal["active", "inactive", "pending"]
    
    @validator('email')
    def validate_email(cls, v):
        return Email(v.lower())

# ✅ Type-safe dataclass
@dataclass(frozen=True)
class Config:
    api_url: str
    timeout: int = Field(gt=0)
```

## PyLance Error Solutions

1. **"Argument missing"**: Use overloads or proper defaults
2. **"Cannot assign to method"**: Use delegation/composition
3. **"Incompatible return"**: Return complete, valid data
4. **Debug with**: `reveal_type()`, intermediate annotations, cast with runtime check

## Runtime Validation

```python
from typeguard import typechecked
from beartype import beartype

@typechecked  # Or @beartype for zero-overhead
def process_user(user_data: UserData) -> ProcessedUser:
    return ProcessedUser(...)
```

## Quality Gates
- `mypy --strict` passes
- PyLance no errors
- No Any/object/broad types
- All tests pass including type validation
- Business constraints in types

**Core principle**: Type widening is technical debt. Any/ignore = failure.

---

# Project Structure Guidelines

## File Size Limits

**Thresholds:**
- **> 200 lines**: Review for separation
- **> 400 lines**: Consider splitting
- **> 600 lines**: Plan refactoring
- **> 1000 lines**: STOP - refactor immediately
