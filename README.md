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

To run a prompt through `grok-3-latest` (default model):

```bash
llm -m grok-3-latest 'What is the meaning of life, the universe, and everything?'
```

To start an interactive chat session:

```bash
llm chat -m grok-3-latest
```

Example chat session:
```
Chatting with grok-3-latest
Type 'exit' or 'quit' to exit
Type '!multi' to enter multiple lines, then '!end' to finish
> Tell me a joke about programming
```

To use a system prompt to give Grok specific instructions:

```bash
cat example.py | llm -m grok-3-latest -s 'explain this code in a humorous way'
```

## Available Models

The following Grok models are available:

- `grok-3-latest` (default)
- `grok-3-mini-fast-latest`
- `grok-3-mini-latest`
- `grok-3-fast-latest`
- `grok-2-latest`
- `grok-2-vision-latest`

You can check the available models using:
```bash
llm grok models
```

## Model Options

The grok-3-latest model accepts the following options, using `-o name value` syntax:

* `-o temperature 0.7`: The sampling temperature, between 0 and 1. Higher values like 0.8 increase randomness, while lower values like 0.2 make the output more focused and deterministic.
* `-o max_tokens 100`: Maximum number of tokens to generate in the completion.

Example with options:

```bash
llm -m grok-3-latest -o temperature 0.2 -o max_tokens 50 'Write a haiku about AI'
```

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
pytest
```

## Available Commands

List available Grok models:
```bash
llm grok models
```

## API Documentation

This plugin uses the xAI API. For more information about the API, see:
- [xAI API Documentation](https://docs.x.ai/api/integrations#openai-sdk)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

Apache License 2.0
