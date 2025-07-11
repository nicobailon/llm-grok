# llm-grok Architecture

This document describes the architecture and design of the `llm-grok` plugin for the LLM CLI tool. The plugin provides access to xAI's Grok models with full support for multimodal capabilities, function calling, and advanced features.

## Overview

The `llm-grok` plugin provides two implementation approaches to suit different needs:

1. **Consolidated Implementation** (`llm_grok.py`) - A single-file implementation that's simple, maintainable, and includes all core features
2. **Modular Implementation** - A multi-file architecture with enterprise features for complex deployments

## Implementation Comparison

### Consolidated Implementation (Recommended)

**File**: `llm_grok/llm_grok.py` (~450 lines)

**Architecture**:
```
llm_grok.py
├── Model Registry (MODELS dict)
├── Exception Classes (3 types)
├── HTTP Client (GrokClient)
├── Main Model Class (Grok)
├── Plugin Registration
└── CLI Commands
```

**Features**:
- All Grok models (2, 3, and 4)
- Streaming responses
- Image/vision support
- Function calling
- Simple retry logic
- Basic error handling

**Benefits**:
- Easy to understand and modify
- Minimal dependencies
- Quick debugging
- Simple deployment
- Ideal for personal projects and MVPs

### Modular Implementation

**Directory Structure**:
```
llm_grok/
├── __init__.py           # Plugin entry point
├── client.py             # HTTP client with advanced features
├── exceptions.py         # Exception hierarchy (8 types)
├── types.py              # Comprehensive type definitions
├── models.py             # Model registry and utilities
├── formats/              # API format handling
│   ├── base.py          # Abstract base handler
│   ├── openai.py        # OpenAI format conversions
│   └── anthropic.py     # Anthropic format conversions
├── processors/           # Content processors
│   ├── multimodal.py    # Image processing
│   ├── tools.py         # Function calling
│   └── streaming.py     # SSE stream parsing
└── utils.py             # Utility functions
```

**Advanced Features**:
- Connection pooling with lifecycle management
- Circuit breaker pattern for fault tolerance
- SSRF protection and URL validation
- Comprehensive type safety with TypedDict
- Abstract base classes for extensibility
- Thread-safe shared resources
- Request size validation and caching
- Exponential backoff with jitter

**Benefits**:
- Enterprise-grade reliability
- Extensible architecture
- Comprehensive error handling
- Performance optimizations
- Suitable for production deployments

## Data Flow

### Consolidated Implementation Flow

```
User Input → LLM CLI → Grok.execute()
                             ↓
                    Build Messages Array
                             ↓
                    Process Images (if any)
                             ↓
                    Create HTTP Client
                             ↓
                    Make API Request (with retry)
                             ↓
                    Parse Response
                             ↓
                    Return to User
```

### Modular Implementation Flow

```
User Input → LLM CLI → Grok Model Instance
                             ↓
                    Shared Client Pool
                             ↓
                    Format Detection & Conversion
                             ↓
                    Content Processors
                             ↓
                    HTTP Client (Circuit Breaker)
                             ↓
                    Stream/Response Processing
                             ↓
                    Format Conversion
                             ↓
                    Return to User
```

## Key Design Decisions

### Consolidated Implementation

1. **Single File**: All code in one place for easy navigation
2. **Simple Types**: Basic dictionaries instead of TypedDict
3. **Direct Processing**: No abstract classes or processors
4. **Basic Retry**: Simple exponential backoff without jitter
5. **Minimal Exceptions**: Only 3 exception types needed

### Modular Implementation

1. **Separation of Concerns**: Each module has a single responsibility
2. **Type Safety**: Full TypedDict definitions for all API types
3. **Extensibility**: Abstract base classes and processors
4. **Fault Tolerance**: Circuit breaker and connection pooling
5. **Security**: SSRF protection and validation layers

## When to Use Each Implementation

### Use Consolidated Implementation When:
- Building personal projects or MVPs
- Simplicity is more important than features
- You need to understand/modify the code quickly
- Running in resource-constrained environments
- You don't need enterprise features

### Use Modular Implementation When:
- Building production applications
- Multiple developers working on the codebase
- Need extensive error handling and recovery
- Require connection pooling for performance
- Security features like SSRF protection are critical

## Performance Characteristics

### Consolidated Implementation
- **Startup**: Fast (single file import)
- **Memory**: Minimal footprint
- **Connections**: New connection per request
- **Error Recovery**: Basic retry logic

### Modular Implementation
- **Startup**: Slower (multiple imports)
- **Memory**: Higher due to connection pool
- **Connections**: Reused from pool
- **Error Recovery**: Circuit breaker + retry

## Extension Points

### Adding a New Model

**Consolidated**:
```python
MODELS["new-model"] = {
    "context_window": 128000,
    "supports_vision": True,
    "supports_tools": False,
}
```

**Modular**:
```python
# In models.py
MODEL_INFO["new-model"] = {
    "context_window": 128000,
    "supports_vision": True,
    "supports_tools": False,
    "pricing_tier": "standard",
    "max_output_tokens": 4096,
}
```

### Adding a New Feature

**Consolidated**: Add method directly to Grok class

**Modular**: Create new processor in `processors/` directory

## Migration Path

If you start with the consolidated implementation and need to migrate to modular:

1. The consolidated implementation uses the same API and patterns
2. Response format is identical between implementations
3. Configuration and usage remain the same
4. Simply switch the import in `__init__.py`

## Testing Strategy

### Consolidated Implementation
- Single test file testing all functionality
- Mock HTTP responses
- Focus on integration testing

### Modular Implementation
- Unit tests for each module
- Integration tests for workflows
- Type checking with mypy
- Coverage requirements

## Conclusion

The dual implementation approach provides flexibility:
- Start simple with the consolidated version
- Migrate to modular when complexity demands it
- Both implementations share the same external API
- Choose based on your specific requirements