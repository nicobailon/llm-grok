"""Streaming response processing."""
import json
import logging
from collections.abc import Iterator
from typing import Any, Optional

from ..constants import (
    DEFAULT_ENCODING,
    FIRST_CHOICE_INDEX,
)
from ..formats import AnthropicFormatHandler, OpenAIFormatHandler
from ..types import StreamEvent
from llm_grok.processors import ContentProcessor, ProcessorConfig

logger = logging.getLogger(__name__)


class StreamProcessor(ContentProcessor[Iterator[bytes], Iterator[dict[str, Any]]]):
    """Handles streaming response processing."""

    def __init__(self, model_id: str, config: Optional[ProcessorConfig] = None):
        """Initialize stream processor."""
        self.model_id = model_id
        self._config = config or ProcessorConfig()
        self._openai_formatter = OpenAIFormatHandler(model_id)
        self._anthropic_formatter = AnthropicFormatHandler(model_id)

    @property
    def config(self) -> ProcessorConfig:
        """Get processor configuration."""
        return self._config

    def process(self, content: Iterator[bytes]) -> Iterator[dict[str, Any]]:
        """Process raw byte stream and yield structured events.

        Handles both OpenAI and Anthropic SSE formats, parsing the stream
        and emitting typed events for content, tool calls, errors, and completion.

        Args:
            content: Iterator of raw bytes from HTTP response

        Yields:
            Structured event dictionaries with type and data

        Raises:
            ValueError: If buffer exceeds maximum size
        """
        stream = content
        buffer = ""
        from ..client import GrokClient
        max_buffer_size = GrokClient.MAX_BUFFER_SIZE

        # Track accumulated state
        accumulated_content = ""
        accumulated_tool_calls: list[dict[str, Any]] = []

        for chunk in stream:
            if chunk:
                # Decode chunk and check buffer size
                chunk_str = chunk.decode(DEFAULT_ENCODING, errors="replace")
                if len(buffer) + len(chunk_str) > max_buffer_size:
                    yield {
                        "type": "error",
                        "error": f"Stream buffer exceeded {max_buffer_size} bytes"
                    }
                    return

                buffer += chunk_str

                # Try to parse as OpenAI format first
                while True:
                    parsed_data, remaining_buffer = self._openai_formatter.parse_openai_sse(buffer)
                    if parsed_data is None:
                        break

                    # Update buffer with remaining content after successful parse
                    buffer = remaining_buffer

                    if parsed_data.get("done"):
                        yield {"type": "done", "data": {}}
                        return

                    # Process OpenAI format chunk
                    if "choices" in parsed_data and parsed_data["choices"]:
                        choice = parsed_data["choices"][FIRST_CHOICE_INDEX]
                        delta = choice.get("delta", {})

                        # Emit content event
                        if "content" in delta and delta["content"]:
                            accumulated_content += delta["content"]
                            yield {
                                "type": "content",
                                "data": {"text": delta["content"]}
                            }

                        # Accumulate tool calls
                        if "tool_calls" in delta:
                            for tool_call in delta["tool_calls"]:
                                self._accumulate_streaming_tool_call(
                                    accumulated_tool_calls, tool_call
                                )

                        # Check for finish
                        if choice.get("finish_reason"):
                            # Emit final tool calls if any
                            if accumulated_tool_calls:
                                yield {
                                    "type": "tool_calls",
                                    "data": {"calls": accumulated_tool_calls}
                                }
                            yield {"type": "done", "data": {}}
                            return

        # Handle any remaining data
        if accumulated_content or accumulated_tool_calls:
            if accumulated_tool_calls:
                yield {
                    "type": "tool_calls",
                    "data": {"calls": accumulated_tool_calls}
                }
            yield {"type": "done", "data": {}}

    def validate(self, event: StreamEvent) -> bool:
        """Validate a streaming event structure.

        Args:
            event: Streaming event to validate

        Returns:
            True if event is valid

        Raises:
            ValueError: If event structure is invalid
        """
        if not isinstance(event, dict):
            raise ValueError("Event must be a dictionary")

        if "type" not in event:
            raise ValueError("Event missing 'type' field")

        event_type = event["type"]
        valid_types = {"content", "tool_calls", "error", "done"}

        if event_type not in valid_types:
            raise ValueError(f"Invalid event type: {event_type}")

        # Validate event data based on type
        if event_type == "content":
            if "data" not in event or "text" not in event["data"]:
                raise ValueError("Content event must have data.text")

        elif event_type == "tool_calls":
            if "data" not in event or "calls" not in event["data"]:
                raise ValueError("Tool calls event must have data.calls")
            if not isinstance(event["data"]["calls"], list):
                raise ValueError("Tool calls must be a list")

        elif event_type == "error":
            if "error" not in event:
                raise ValueError("Error event must have error field")

        return True

    def _accumulate_streaming_tool_call(
        self, accumulator: list[dict[str, Any]], delta: dict[str, Any]
    ) -> None:
        """Accumulate incremental tool call data from stream.

        Args:
            accumulator: List to accumulate tool calls into
            delta: Incremental tool call data
        """
        index = delta.get("index", FIRST_CHOICE_INDEX)

        # Ensure accumulator has enough entries
        while len(accumulator) <= index:
            accumulator.append({
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""}
            })

        tool_call = accumulator[index]

        # Merge incremental data
        if "id" in delta:
            tool_call["id"] = delta["id"]

        if "type" in delta:
            tool_call["type"] = delta["type"]

        if "function" in delta and delta["function"]:
            func_delta = delta["function"]
            if "name" in func_delta:
                tool_call["function"]["name"] = func_delta["name"]
            if "arguments" in func_delta:
                # Accumulate arguments string
                if "arguments" not in tool_call["function"]:
                    tool_call["function"]["arguments"] = ""
                tool_call["function"]["arguments"] += func_delta["arguments"]

    def process_stream(self, http_response: Any, response: Any, use_messages: bool) -> Iterator[str]:
        """Process streaming HTTP response and yield content.

        Args:
            http_response: The streaming HTTP response
            response: The LLM response object to accumulate tool calls
            use_messages: Whether this is using the Anthropic messages endpoint

        Yields:
            Content strings as they arrive
        """
        buffer = ""
        # Import MAX_BUFFER_SIZE from client
        from ..client import GrokClient
        max_buffer_size = GrokClient.MAX_BUFFER_SIZE

        for chunk in http_response.iter_raw():
            if chunk:
                # Check buffer size before appending
                chunk_str = chunk.decode(DEFAULT_ENCODING)
                if len(buffer) + len(chunk_str) > max_buffer_size:
                    raise ValueError(
                        f"Streaming buffer exceeded maximum size of {max_buffer_size} bytes. "
                        "Response is too large to process safely."
                    )
                buffer += chunk_str

                if use_messages:
                    # Anthropic SSE format parsing
                    while True:
                        result, buffer = self._anthropic_formatter.parse_anthropic_sse(buffer)
                        if result is None:
                            break

                        event_type, event_data = result
                        # Convert Anthropic event to OpenAI format
                        openai_chunk = self._anthropic_formatter.convert_anthropic_stream_chunk(event_type, event_data)
                        if openai_chunk and "choices" in openai_chunk and openai_chunk["choices"]:
                            choice = openai_chunk["choices"][FIRST_CHOICE_INDEX]
                            delta = choice.get("delta", {})
                            content = self._process_stream_delta(dict(delta), response)
                            if content:
                                yield content
                else:
                    # OpenAI SSE format parsing
                    while True:
                        parsed_data, buffer = self._openai_formatter.parse_openai_sse(buffer)
                        if parsed_data is None:
                            break
                        if parsed_data.get("done"):
                            break

                        if "choices" in parsed_data and parsed_data["choices"]:
                            choice = parsed_data["choices"][FIRST_CHOICE_INDEX]
                            delta = choice.get("delta", {})
                            content = self._process_stream_delta(delta, response)
                            if content:
                                yield content

        # Finalize tool calls after streaming completes
        self._finalize_tool_calls(response)

    def _process_stream_delta(self, delta: dict[str, Any], response: Any) -> Optional[str]:
        """Process a stream delta and return content to yield, if any."""
        content_to_yield = None

        # Handle streaming content
        if "content" in delta:
            content = delta["content"]
            if content:
                content_to_yield = content

        # Handle streaming tool calls
        if "tool_calls" in delta:
            for tool_call in delta["tool_calls"]:
                self._accumulate_tool_call(response, tool_call)

        return content_to_yield

    def _accumulate_tool_call(self, response: Any, tool_call: dict[str, Any]) -> None:
        """Helper to accumulate streaming tool call data."""
        # Initialize accumulator if not exists
        if not hasattr(response, '_tool_calls_accumulator'):
            response._tool_calls_accumulator = []

        if tool_call.get("index") is not None:
            index = tool_call["index"]
            # Ensure list is large enough
            tool_calls_accumulator = getattr(response, '_tool_calls_accumulator', [])
            while len(tool_calls_accumulator) <= index:
                tool_calls_accumulator.append({})

            # Merge tool call data
            if "id" in tool_call:
                tool_calls_accumulator[index]["id"] = tool_call["id"]
            if "type" in tool_call:
                tool_calls_accumulator[index]["type"] = tool_call["type"]
            if "function" in tool_call:
                if "function" not in tool_calls_accumulator[index]:
                    tool_calls_accumulator[index]["function"] = {}
                if "name" in tool_call["function"]:
                    tool_calls_accumulator[index]["function"]["name"] = tool_call["function"]["name"]
                if "arguments" in tool_call["function"]:
                    if "arguments" not in tool_calls_accumulator[index]["function"]:
                        tool_calls_accumulator[index]["function"]["arguments"] = ""
                    tool_calls_accumulator[index]["function"]["arguments"] += tool_call["function"]["arguments"]

    def _finalize_tool_calls(self, response: Any) -> None:
        """Convert accumulated tool calls to proper llm.ToolCall objects."""
        if hasattr(response, '_tool_calls_accumulator') and getattr(response, '_tool_calls_accumulator', None):
            # Import here to avoid circular dependency
            import llm

            # Check if this is a real llm.Response or a mock
            if hasattr(response, 'add_tool_call'):
                # Real llm.Response - use the proper method
                tool_calls_accumulator = getattr(response, '_tool_calls_accumulator', [])
                for tool_call_data in tool_calls_accumulator:
                    if tool_call_data.get("function") and tool_call_data["function"].get("name"):
                        try:
                            # Parse the accumulated arguments JSON
                            arguments = json.loads(tool_call_data["function"].get("arguments", "{}"))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse accumulated tool call arguments: {e}, using empty dict")
                            arguments = {}

                        # Add the tool call using the proper method
                        response.add_tool_call(
                            llm.ToolCall(
                                tool_call_id=tool_call_data.get("id"),
                                name=tool_call_data["function"]["name"],
                                arguments=arguments
                            )
                        )
            else:
                # MockResponse or similar - store raw format
                # Use setattr to set tool_calls attribute
                response.tool_calls = getattr(response, '_tool_calls_accumulator', [])

            # Clean up the accumulator
            delattr(response, '_tool_calls_accumulator')
