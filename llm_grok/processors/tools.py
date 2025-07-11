"""Tool calling and function execution processing."""
import json
import logging
from typing import Optional, Union, TypeGuard, Protocol, runtime_checkable, Dict, cast, Any

from llm_grok.processors import ContentProcessor, ProcessorConfig, ValidationError

from ..types import AnthropicResponse, OpenAIResponse, RawFunctionCall, RawToolCall, ToolCall, ToolCallWithIndex


@runtime_checkable
class LLMResponseProtocol(Protocol):
    """Protocol for LLM framework response objects."""
    
    def tool_calls(self) -> list[ToolCall]: ...
    def add_tool_call(self, tool_call: Any) -> None: ...


@runtime_checkable
class MockResponseProtocol(Protocol):
    """Protocol for mock response objects used in testing."""
    tool_calls: list[ToolCall]


def has_add_tool_call(obj: object) -> TypeGuard[LLMResponseProtocol]:
    """Type guard for objects with add_tool_call method."""
    return hasattr(obj, 'add_tool_call') and callable(getattr(obj, 'add_tool_call'))


def has_tool_calls_attr(obj: object) -> TypeGuard[MockResponseProtocol]:
    """Type guard for objects with tool_calls attribute."""
    return hasattr(obj, 'tool_calls')

logger = logging.getLogger(__name__)


class ToolProcessor(ContentProcessor[Union[OpenAIResponse, AnthropicResponse], list[ToolCall]]):
    """Handles tool calling and function execution.
    
    Processes function/tool calls from API responses, supporting both OpenAI and Anthropic
    formats. Handles validation, format conversion, streaming accumulation, and parallel
    tool call execution.
    
    Key features:
    - Unified handling of OpenAI and Anthropic tool call formats
    - Streaming tool call accumulation for real-time processing
    - Parallel tool call support
    - Robust JSON argument validation and error recovery
    """

    def __init__(self, model_id: str, config: Optional[ProcessorConfig] = None):
        """Initialize tool processor."""
        self.model_id = model_id
        self._config = config or ProcessorConfig()

    @property
    def config(self) -> ProcessorConfig:
        """Get processor configuration."""
        return self._config

    def _is_openai_response(self, response: Union[OpenAIResponse, AnthropicResponse, Dict[str, object]]) -> TypeGuard[OpenAIResponse]:
        """Type guard for OpenAI response format."""
        return isinstance(response, dict) and "choices" in response and isinstance(response.get("choices"), list)

    def _is_anthropic_response(self, response: Union[OpenAIResponse, AnthropicResponse, Dict[str, object]]) -> TypeGuard[AnthropicResponse]:
        """Type guard for Anthropic response format."""
        return isinstance(response, dict) and "content" in response and isinstance(response.get("content"), list)

    def process(self, content: Union[OpenAIResponse, AnthropicResponse, Dict[str, object]]) -> list[ToolCall]:
        """Process tool calls from API response.
        
        Extracts and validates tool calls from both OpenAI and Anthropic response formats,
        handling both single and parallel tool calls.
        
        Args:
            content: API response containing tool calls
            
        Returns:
            List of validated and formatted ToolCall objects
            
        Raises:
            ValidationError: If tool calls are malformed or invalid
        """
        if not isinstance(content, dict):
            return []
        
        # Use type guards to narrow the type
        if self._is_openai_response(content):
            return self._extract_openai_tool_calls(content)
        elif self._is_anthropic_response(content):
            return self._extract_anthropic_tool_calls(content)
        return []

    def _extract_openai_tool_calls(self, response: OpenAIResponse) -> list[ToolCall]:
        """Extract tool calls from OpenAI response."""
        tool_calls: list[ToolCall] = []
        
        if response["choices"]:
            choice = response["choices"][0]
            message = choice["message"]
            
            if "tool_calls" in message and message["tool_calls"]:
                for raw_call in message["tool_calls"]:
                    # Use proper typing for tool call validation
                    validated_call = self._validate_and_format_tool_call(cast(RawToolCall, raw_call))
                    if validated_call:
                        tool_calls.append(validated_call)
        
        return tool_calls

    def _extract_anthropic_tool_calls(self, response: AnthropicResponse) -> list[ToolCall]:
        """Extract tool calls from Anthropic response."""
        tool_calls: list[ToolCall] = []
        
        for content_block in response["content"]:
            if content_block.get("type") == "tool_use":
                # Convert Anthropic format to OpenAI format
                tool_call: ToolCall = {
                    "id": str(content_block.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": str(content_block.get("name", "")),
                        "arguments": json.dumps(content_block.get("input", {}))
                    }
                }
                tool_calls.append(tool_call)

        return tool_calls

    def validate(self, tool_calls: list[ToolCall]) -> bool:
        """Validate a list of tool calls.
        
        Args:
            tool_calls: List of tool calls to validate
            
        Returns:
            True if all tool calls are valid
            
        Raises:
            ValidationError: If any tool call is invalid
        """
        if not isinstance(tool_calls, list):
            raise ValidationError("Tool calls must be a list")

        for i, call in enumerate(tool_calls):
            if not isinstance(call, dict):
                raise ValidationError(f"Tool call {i} must be a dict")

            # Validate required fields
            if "id" not in call:
                raise ValidationError(f"Tool call {i} missing 'id' field")
            if "type" not in call:
                raise ValidationError(f"Tool call {i} missing 'type' field")
            if "function" not in call:
                raise ValidationError(f"Tool call {i} missing 'function' field")

            # Validate function structure
            func = call["function"]
            if not isinstance(func, dict):
                raise ValidationError(f"Tool call {i} function must be a dict")
            if "name" not in func:
                raise ValidationError(f"Tool call {i} function missing 'name' field")
            if "arguments" not in func:
                raise ValidationError(f"Tool call {i} function missing 'arguments' field")

            # Validate arguments are valid JSON string
            if isinstance(func["arguments"], str):
                try:
                    json.loads(func["arguments"])
                except json.JSONDecodeError:
                    raise ValidationError(
                        f"Tool call {i} has invalid JSON in arguments: {func['arguments']}"
                    )
            else:
                raise ValidationError(f"Tool call {i} arguments must be a JSON string")

        return True

    def _validate_and_format_tool_call(self, raw_call: RawToolCall) -> Optional[ToolCallWithIndex]:
        """Validate and format a single tool call.
        
        Args:
            raw_call: Raw tool call data from API
            
        Returns:
            Formatted ToolCall or None if invalid
        """
        try:
            # Extract required fields
            call_id = raw_call.get("id")
            call_type = raw_call.get("type", "function")
            function_data = raw_call.get("function")

            if not call_id or not function_data:
                return None

            # Type narrowed by existence check above
            func_data = function_data
            name = func_data.get("name")
            arguments_str = func_data.get("arguments", "{}")

            if not name:
                return None

            # Validate arguments JSON
            try:
                if arguments_str:
                    json.loads(arguments_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse tool arguments: {e}, using empty dict")
                arguments_str = "{}"

            # Create formatted tool call
            tool_call: ToolCallWithIndex = {
                "id": call_id,
                "type": "function",  # Always function type for tool calls
                "function": {
                    "name": name,
                    "arguments": arguments_str
                }
            }

            # Add optional index
            if "index" in raw_call:
                tool_call["index"] = raw_call["index"]

            return tool_call

        except Exception as e:
            logger.warning(f"Failed to convert tool call: {e}, skipping")
            return None

    def process_tool_calls(self, response: object, tool_calls: list[ToolCall]) -> None:
        """Process and add tool calls to the response object.
        
        Args:
            response: The LLM response object (real llm.Response or mock for testing)
            tool_calls: List of properly typed tool calls from the API
        """
        # Import here to avoid circular dependency
        import llm

        if has_add_tool_call(response):
            # Real llm.Response - convert to proper llm.ToolCall objects
            for tool_call in tool_calls:
                function_details = tool_call["function"]
                try:
                    arguments = json.loads(function_details["arguments"])
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse tool use input: {e}, using empty dict")
                    arguments = {}

                # Create llm.ToolCall object for the framework
                llm_tool_call = llm.ToolCall(
                    tool_call_id=tool_call["id"],
                    name=function_details["name"],
                    arguments=arguments
                )
                response.add_tool_call(llm_tool_call)
        elif has_tool_calls_attr(response):
            # MockResponse - store raw format for testing
            response.tool_calls = tool_calls

    def accumulate_tool_call(self, accumulator: list[RawToolCall], delta: RawToolCall) -> None:
        """Accumulate streaming tool call chunks.
        
        Args:
            accumulator: List to accumulate tool calls into
            delta: The incremental tool call data from the stream
        """
        index = delta.get("index", 0)

        # Ensure accumulator has enough entries
        while len(accumulator) <= index:
            accumulator.append(cast(RawToolCall, {
                "index": len(accumulator),
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""}
            }))

        tool_call = accumulator[index]

        # Merge tool call data
        if "id" in delta:
            tool_call["id"] = delta["id"]

        if "type" in delta:
            tool_call["type"] = delta["type"]

        if "function" in delta:
            func = delta["function"]
            tool_func = tool_call.get("function")
            if tool_func and func and "name" in func:
                tool_func["name"] = func["name"]
            if tool_func and func and "arguments" in func:
                # Accumulate arguments (they come in chunks)
                if "arguments" not in tool_func:
                    tool_func["arguments"] = ""
                tool_func["arguments"] += func["arguments"]

    def finalize_tool_calls(self, accumulated: list[RawToolCall]) -> list[ToolCallWithIndex]:
        """Convert accumulated tool calls to final format.
        
        Args:
            accumulated: List of accumulated tool call data
            
        Returns:
            List of properly formatted ToolCallWithIndex objects
        """
        finalized: list[ToolCallWithIndex] = []

        for tool_call in accumulated:
            # Skip incomplete tool calls
            function_data = tool_call.get("function")
            if not tool_call.get("id") or not function_data or not function_data.get("name"):
                continue

            # Type narrowed by validation above
            func_data = function_data

            # Parse arguments JSON
            try:
                arguments_str = func_data.get("arguments", "{}")
                arguments = json.loads(arguments_str) if arguments_str else {}
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse accumulated arguments: {e}, using empty dict")
                arguments = {}

            # Create final ToolCall
            finalized_call: ToolCallWithIndex = {
                "id": tool_call.get("id", ""),
                "type": "function",  # Always function type for tool calls
                "function": {
                    "name": func_data.get("name", ""),
                    "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments_str
                }
            }

            # Add index if present
            if "index" in tool_call:
                finalized_call["index"] = tool_call["index"]

            finalized.append(finalized_call)

        return finalized

    def validate_tool_definition(self, tool: Dict[str, object]) -> bool:
        """Validate tool conforms to expected schema.
        
        Args:
            tool: Tool definition to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check required top-level fields
        if not isinstance(tool, dict):
            return False

        if tool.get("type") != "function":
            return False

        # Check function definition
        function = tool.get("function")
        if not isinstance(function, dict):
            return False

        # Check required function fields
        required_fields = ["name", "description", "parameters"]
        if not all(field in function for field in required_fields):
            return False

        # Validate name
        if not isinstance(function["name"], str) or not function["name"]:
            return False

        # Validate description
        if not isinstance(function["description"], str):
            return False

        # Validate parameters schema
        parameters = function["parameters"]
        if not isinstance(parameters, dict):
            return False

        # Basic JSON schema validation
        if parameters.get("type") != "object":
            return False

        if "properties" not in parameters:
            return False

        return True
