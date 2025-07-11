# llm-grok

[![PyPI](https://img.shields.io/pypi/v/llm-grok.svg)](https://pypi.org/project/llm-grok/)
[![Tests](https://github.com/hiepler/llm-grok/workflows/Test/badge.svg)](https://github.com/hiepler/llm-grok/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/hiepler/llm-grok/blob/main/LICENSE)

Plugin for [LLM](https://llm.datasette.io/) providing access to Grok models using the xAI API

## Installation

Install this plugin in the same environment as LLM:

```bash
llm install llm-grok
```

## Usage

First, obtain an API key from xAI.

Configure the key using the `llm keys set grok` command:

```bash
llm keys set grok
# Paste your xAI API key here
```

You can also set it via environment variable:
```bash
export XAI_API_KEY="your-api-key-here"
```

You can now access the Grok model. Run `llm models` to see it in the list.

To run a prompt through Grok 4 (default model):

```bash
llm -m x-ai/grok-4 'What is the meaning of life, the universe, and everything?'
```

To start an interactive chat session:

```bash
llm chat -m x-ai/grok-4
```

Example chat session:
```
Chatting with x-ai/grok-4
Type 'exit' or 'quit' to exit
Type '!multi' to enter multiple lines, then '!end' to finish
> Tell me a joke about programming
```

To use a system prompt to give Grok specific instructions:

```bash
cat example.py | llm -m x-ai/grok-4 -s 'explain this code in a humorous way'
```

To set your default model:

```bash
llm models default x-ai/grok-4
# Now running `llm ...` will use `x-ai/grok-4` by default
```

## Available Models

The following Grok models are available:

### Grok 4 Models (NEW)
- `x-ai/grok-4` (default) - Most capable model with 256k context
- `grok-4-heavy` - Extended reasoning capabilities

### Grok 3 Models
- `grok-3-latest` - Previous generation flagship
- `grok-3-fast-latest` - Faster variant 
- `grok-3-mini-latest` - Smaller, efficient model
- `grok-3-mini-fast-latest` - Fastest mini variant

### Grok 2 Models
- `grok-2-latest` - Legacy model
- `grok-2-vision-latest` - Vision-capable model

You can check the available models using:
```bash
llm grok models
```

## Model Options

All Grok models accept the following options, using `-o name value` syntax:

* `-o temperature 0.7`: The sampling temperature, between 0 and 1. Higher values like 0.8 increase randomness, while lower values like 0.2 make the output more focused and deterministic.
* `-o max_completion_tokens 100`: Maximum number of tokens to generate in the completion (includes both visible tokens and reasoning tokens).
* `-o use_messages_endpoint true`: Use the alternative `/messages` endpoint (Anthropic-compatible format)
* `-o reasoning_effort <level>`: Control reasoning depth for Grok 4 models

Example with options:

```bash
llm -m x-ai/grok-4 -o temperature 0.2 -o max_completion_tokens 50 'Write a haiku about AI'
```

## Advanced Features

### Multimodal Support (Grok 4 & Grok 2 Vision)

Analyze images with vision-capable models:

```bash
# Analyze a local image
llm -m x-ai/grok-4 'What do you see in this image?' -a image.jpg

# Analyze an image from URL
llm -m grok-2-vision-latest 'Describe this chart' -a https://example.com/chart.png
```

### Function Calling (Grok 4)

Grok 4 models support OpenAI-compatible function calling for tool use:

```python
import llm

model = llm.get_model("x-ai/grok-4")
response = model.prompt(
    "What's the weather in New York?",
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }]
)
```

### Alternative Endpoints

The plugin supports both the standard OpenAI-compatible endpoint and an alternative Anthropic-compatible `/messages` endpoint:

```bash
# Use the messages endpoint (better for complex conversations)
llm -m x-ai/grok-4 -o use_messages_endpoint true 'Hello, how are you?'
```

#### Endpoint Differences

| Feature | `/chat/completions` (default) | `/messages` |
|---------|------------------------------|-------------|
| Format | OpenAI-compatible | Anthropic-compatible |
| System Messages | In messages array | Separate `system` parameter |
| Max Tokens | `max_completion_tokens` | `max_tokens` |
| Tool Calls | OpenAI format | Anthropic `tool_use` format |
| Streaming | SSE with `data:` prefix | SSE with `event:` and `data:` |
| Response Format | Supported | Not supported |

The messages endpoint provides better handling of:
- Multiple system messages (automatically combined)
- Complex tool interactions
- Streaming with detailed event types

## Architecture

The llm-grok plugin uses a clean, modular architecture optimized for reliability and maintainability:

### 🏗️ **Component-Based Design**
- **Processors**: Handle content transformation (multimodal, streaming, tools)
- **Formatters**: Convert between API formats (OpenAI ↔ Anthropic)  
- **Client**: Enterprise HTTP client with retry logic and connection pooling
- **Models**: Centralized model capability registry

### 🎓 **Learning Resources**
For understanding the core concepts, see the simplified reference implementation at `examples/consolidated_plugin.py` (~500 lines).

### 🔧 **Enterprise Features**
- Thread-safe shared resource management
- Circuit breakers and connection pooling
- Comprehensive error handling hierarchy
- SSRF protection and security measures
- Type-safe API with full TypedDict coverage

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:

```bash
git clone https://github.com/hiepler/llm-grok.git
cd llm-grok
python3 -m venv venv
source venv/bin/activate
```

Now install the dependencies and test dependencies:

```bash
pip install -e '.[test]'
```

To run the tests:

```bash
# Run all tests (using pytest - our only test framework)
pytest

# Run with coverage
pytest --cov=llm_grok

# Type checking (generates .mypy_cache/)
mypy llm_grok --strict

# Linting and formatting (generates .ruff_cache/)
ruff check llm_grok/
ruff format llm_grok/
```

### Development Tools

This project uses the following development tools:

- **pytest**: Test framework (cache: `.pytest_cache/`)
- **mypy**: Static type checker (cache: `.mypy_cache/`)  
- **ruff**: Fast linter and formatter (cache: `.ruff_cache/`)

All cache directories are automatically regenerated and are ignored by git. You can safely delete them anytime:

```bash
rm -rf .pytest_cache .mypy_cache .ruff_cache
```

## Available Commands

List available Grok models:
```bash
llm grok models
```

## Troubleshooting

### Common Issues

**Authentication Error**
```
Error: Invalid API key
```
Solution: Ensure your API key is correctly set:
```bash
llm keys set grok
# Or use environment variable:
export XAI_API_KEY="your-key-here"
```

**Rate Limiting**
```
Error: Rate limit exceeded
```
The plugin automatically retries with exponential backoff. For persistent issues:
- Check your usage at https://console.x.ai
- Consider using a model with higher rate limits
- Reduce request frequency

**Image Analysis Fails**
```
Warning: Skipping invalid image
```
Ensure:
- Image format is JPEG, PNG, GIF, or WebP
- URLs are publicly accessible
- Base64 data is properly encoded
- File size is under 10MB

**Connection Issues**
```
Error: Network error
```
The plugin automatically retries failed requests up to 3 times. If issues persist:
- Check your network connection
- Verify firewall settings allow access to api.x.ai
- Try again after a few moments

### Debug Mode

Enable detailed logging:
```bash
# Set log level
export LLM_LOG_LEVEL=DEBUG

# Run with verbose output
llm -m x-ai/grok-4 "test" -v
```

## API Documentation

This plugin uses the xAI API. For more information about the API, see:
- [xAI API Documentation](https://docs.x.ai/docs/overview)

## Technical Details

### Rate Limiting

The plugin handles rate limits automatically:
- Retries up to 3 times with exponential backoff
- Respects `Retry-After` headers from the API
- Clear error messages when limits are exceeded

### Image Support

For vision-capable models (Grok 4, Grok 2 Vision):
- Supports JPEG, PNG, GIF, and WebP formats
- Handles URLs, local files, and base64-encoded data
- Automatic MIME type detection
- Images are converted to base64 data URLs when needed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache License 2.0
