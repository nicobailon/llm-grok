"""Tool calling and function execution processing."""
import json
import logging
from typing import Any, Dict, List, Optional, Union

from llm_grok.processors import ContentProcessor, ValidationError, ProcessorConfig
from ..types import ToolCall, OpenAIResponse, AnthropicResponse

logger = logging.getLogger(__name__)


class ToolProcessor(ContentProcessor[Union[OpenAIResponse, AnthropicResponse, Dict[str, Any]], List[ToolCall]]):
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
    
    def process(self, content: Union[OpenAIResponse, AnthropicResponse, Dict[str, Any]]) -> List[ToolCall]:
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
        response = content
        tool_calls: List[ToolCall] = []
        
        # Handle OpenAI format
        if isinstance(response, dict) and "choices" in response and response["choices"]:
            choice = response["choices"][0]
            message = choice.get("message", {})
            
            if "tool_calls" in message and message["tool_calls"]:
                for raw_call in message["tool_calls"]:
                    validated_call = self._validate_and_format_tool_call(raw_call)
                    if validated_call:
                        tool_calls.append(validated_call)
        
        # Handle Anthropic format
        elif isinstance(response, dict) and "content" in response and isinstance(response["content"], list):
            for content_block in response["content"]:
                if content_block.get("type") == "tool_use":
                    # Convert Anthropic format to OpenAI format
                    openai_call = {
                        "id": content_block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": content_block.get("name", ""),
                            "arguments": json.dumps(content_block.get("input", {}))
                        }
                    }
                    validated_call = self._validate_and_format_tool_call(openai_call)
                    if validated_call:
                        tool_calls.append(validated_call)
        
        return tool_calls
    
    def validate(self, tool_calls: List[ToolCall]) -> bool:
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
    
    def _validate_and_format_tool_call(self, raw_call: Dict[str, Any]) -> Optional[ToolCall]:
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
            function_data = raw_call.get("function", {})
            
            if not call_id or not function_data:
                return None
            
            name = function_data.get("name")
            arguments_str = function_data.get("arguments", "{}")
            
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
            tool_call: ToolCall = {
                "id": call_id,
                "type": call_type,
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
    
    def process_tool_calls(self, response: Any, tool_calls: List[Dict[str, Any]]) -> None:
        """Process and add tool calls to the response object.
        
        Args:
            response: The LLM response object
            tool_calls: List of tool call dictionaries from the API
        """
        # Import here to avoid circular dependency
        import llm
        
        if hasattr(response, 'add_tool_call'):
            # Real llm.Response - convert to proper llm.ToolCall objects
            for tool_call in tool_calls:
                if tool_call.get("function") and tool_call["function"].get("name"):
                    try:
                        arguments = json.loads(tool_call["function"].get("arguments", "{}"))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse tool use input: {e}, using empty dict")
                        arguments = {}
                    
                    response.add_tool_call(
                        llm.ToolCall(
                            tool_call_id=tool_call.get("id"),
                            name=tool_call["function"]["name"],
                            arguments=arguments
                        )
                    )
        else:
            # MockResponse - store raw format
            response.tool_calls = tool_calls
    
    def accumulate_tool_call(self, accumulator: List[Dict[str, Any]], delta: Dict[str, Any]) -> None:
        """Accumulate streaming tool call chunks.
        
        Args:
            accumulator: List to accumulate tool calls into
            delta: The incremental tool call data from the stream
        """
        index = delta.get("index", 0)
        
        # Ensure accumulator has enough entries
        while len(accumulator) <= index:
            accumulator.append({
                "index": len(accumulator),
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""}
            })
        
        tool_call = accumulator[index]
        
        # Merge tool call data
        if "id" in delta:
            tool_call["id"] = delta["id"]
        
        if "type" in delta:
            tool_call["type"] = delta["type"]
        
        if "function" in delta:
            func = delta["function"]
            if "name" in func:
                tool_call["function"]["name"] = func["name"]
            if "arguments" in func:
                # Accumulate arguments (they come in chunks)
                tool_call["function"]["arguments"] += func["arguments"]
    
    def finalize_tool_calls(self, accumulated: List[Dict[str, Any]]) -> List[ToolCall]:
        """Convert accumulated tool calls to final format.
        
        Args:
            accumulated: List of accumulated tool call data
            
        Returns:
            List of properly formatted ToolCall objects
        """
        finalized: List[ToolCall] = []
        
        for tool_call in accumulated:
            # Skip incomplete tool calls
            if not tool_call.get("id") or not tool_call.get("function", {}).get("name"):
                continue
            
            # Parse arguments JSON
            try:
                arguments_str = tool_call["function"].get("arguments", "{}")
                arguments = json.loads(arguments_str) if arguments_str else {}
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse accumulated arguments: {e}, using empty dict")
                arguments = {}
            
            # Create final ToolCall
            finalized_call: ToolCall = {
                "id": tool_call["id"],
                "type": tool_call.get("type", "function"),
                "function": {
                    "name": tool_call["function"]["name"],
                    "arguments": json.dumps(arguments) if isinstance(arguments, dict) else arguments_str
                }
            }
            
            # Add index if present
            if "index" in tool_call:
                finalized_call["index"] = tool_call["index"]
            
            finalized.append(finalized_call)
        
        return finalized
    
    def validate_tool_definition(self, tool: Dict[str, Any]) -> bool:
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