import base64
import json
import sys
import time
from typing import Optional, List, Dict, Union, Any, Iterator, TypedDict, Literal, Type, cast

import click
import httpx
import llm
from pydantic import Field
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Precise type definitions following type safety guidelines
class ImageContent(TypedDict):
    type: Literal["image_url"]
    image_url: Dict[str, str]

class TextContent(TypedDict):
    type: Literal["text"]
    text: str

class FunctionCall(TypedDict, total=False):
    id: str
    type: Literal["function"]
    function: "FunctionCallDetails"

class FunctionCallDetails(TypedDict):
    name: str
    arguments: str  # JSON string

class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: FunctionCallDetails
    index: Optional[int]

class Message(TypedDict, total=False):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[Union[TextContent, ImageContent]]]
    tool_calls: List[ToolCall]

class AnthropicImageSource(TypedDict):
    type: Literal["base64"]
    media_type: str
    data: str

class AnthropicImage(TypedDict):
    type: Literal["image"]
    source: AnthropicImageSource

class AnthropicTextBlock(TypedDict):
    type: Literal["text"]
    text: str

class AnthropicToolUse(TypedDict):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]  # Tool-specific parameters

class AnthropicMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: List[Union[AnthropicTextBlock, AnthropicImage, AnthropicToolUse]]

class AnthropicRequest(TypedDict, total=False):
    messages: List[AnthropicMessage]
    system: str
    model: str
    max_tokens: int
    temperature: float
    tools: List[Dict[str, Any]]
    tool_choice: Union[Literal["auto", "none"], Dict[str, Any]]

class ToolDefinition(TypedDict):
    type: Literal["function"]
    function: Dict[str, Any]  # OpenAI tool schema

class AnthropicToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: Dict[str, Any]  # JSON Schema

AVAILABLE_MODELS = [
    # Grok 4 models
    "x-ai/grok-4",
    "grok-4-heavy",
    # Grok 3 models
    "grok-3-latest",
    "grok-3-fast-latest",
    "grok-3-mini-latest",
    "grok-3-mini-fast-latest",
    # Grok 2 models
    "grok-2-latest",
    "grok-2-vision-latest",
]
DEFAULT_MODEL = "x-ai/grok-4"

# Model capabilities metadata
MODEL_INFO = {
    "x-ai/grok-4": {
        "context_window": 256000,
        "supports_vision": True,
        "supports_tools": True,
        "pricing_tier": "standard",
        "max_output_tokens": 8192,
    },
    "grok-4-heavy": {
        "context_window": 256000,
        "supports_vision": True,
        "supports_tools": True,
        "pricing_tier": "heavy",
        "max_output_tokens": 8192,
    },
    "grok-3-latest": {
        "context_window": 128000,
        "supports_vision": False,
        "supports_tools": False,
        "pricing_tier": "standard",
        "max_output_tokens": 4096,
    },
    "grok-3-fast-latest": {
        "context_window": 128000,
        "supports_vision": False,
        "supports_tools": False,
        "pricing_tier": "standard",
        "max_output_tokens": 4096,
    },
    "grok-3-mini-latest": {
        "context_window": 128000,
        "supports_vision": False,
        "supports_tools": False,
        "pricing_tier": "mini",
        "max_output_tokens": 4096,
    },
    "grok-3-mini-fast-latest": {
        "context_window": 128000,
        "supports_vision": False,
        "supports_tools": False,
        "pricing_tier": "mini",
        "max_output_tokens": 4096,
    },
    "grok-2-latest": {
        "context_window": 32768,
        "supports_vision": False,
        "supports_tools": False,
        "pricing_tier": "standard",
        "max_output_tokens": 4096,
    },
    "grok-2-vision-latest": {
        "context_window": 32768,
        "supports_vision": True,
        "supports_tools": False,
        "pricing_tier": "standard",
        "max_output_tokens": 4096,
    },
}


@llm.hookimpl
def register_models(register: Any) -> None:
    for model_id in AVAILABLE_MODELS:
        register(Grok(model_id))


class GrokError(Exception):
    """Base exception for Grok API errors"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.message = message
        self.details = details
        super().__init__(message)


class RateLimitError(GrokError):
    """Exception for rate limit errors"""

    pass


class QuotaExceededError(GrokError):
    """Exception for quota exceeded errors"""

    pass


class Grok(llm.KeyModel):
    can_stream = True
    needs_key = "grok"
    key_env_var = "XAI_API_KEY"
    MAX_RETRIES = 3
    BASE_DELAY = 1  # Base delay in seconds
    API_URL = "https://api.x.ai/v1/chat/completions"
    MESSAGES_URL = "https://api.x.ai/v1/messages"
    IMAGE_GEN_URL = "https://api.x.ai/v1/image-generations"
    
    class Options(llm.Options):  # type: ignore[assignment]
        temperature: Optional[float] = Field(
                description=(
                    "Determines the sampling temperature. Higher values like 0.8 increase randomness, "
                    "while lower values like 0.2 make the output more focused and deterministic."
                ),
                ge=0,
                le=1,
                default=0.0,
        )
        max_completion_tokens: Optional[int] = Field(
                description="The maximum number of tokens to generate, including visible output tokens and reasoning tokens.",
                ge=0,
                default=None,
        )
        tools: Optional[List[ToolDefinition]] = Field(
                description="List of tool/function definitions in OpenAI format",
                default=None,
        )
        tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
                description="Controls which (if any) function is called. Can be 'auto', 'none', or a specific function",
                default=None,
        )
        response_format: Optional[Dict[str, Any]] = Field(
                description="Structured output format (e.g., {'type': 'json_object'})",
                default=None,
        )
        reasoning_effort: Optional[str] = Field(
                description="Level of reasoning effort for the model",
                default=None,
        )
        use_messages_endpoint: Optional[bool] = Field(
                description="Use Anthropic-style /messages endpoint instead of /chat/completions",
                default=False,
        )

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
    
    def _get_model_capability(self, capability: str) -> bool:
        """Check if current model supports a specific capability.
        
        Args:
            capability: The capability to check (e.g., 'supports_vision', 'supports_tools')
            
        Returns:
            bool: True if the model supports the capability, False otherwise
        """
        return MODEL_INFO.get(self.model_id, {}).get(capability, False)
    
    def _accumulate_tool_call(self, response: llm.Response, tool_call: Dict[str, Any]) -> None:
        """Helper to accumulate streaming tool call data.
        
        Args:
            response: The response object to accumulate tool calls into
            tool_call: The incremental tool call data from the stream
        """
        # Initialize accumulator if not exists
        if not hasattr(response, '_tool_calls_accumulator'):
            setattr(response, '_tool_calls_accumulator', [])
        
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

    def _validate_image_format(self, data: str) -> str:
        """Validate and format image data for multimodal API requests.
        
        Accepts image data in three formats:
        1. HTTP/HTTPS URLs - returned as-is
        2. Data URLs with base64 encoding - validated and returned
        3. Raw base64 strings - MIME type detected and formatted as data URL
        
        Args:
            data (str): The image data as URL, data URL, or base64 string
            
        Returns:
            str: Properly formatted image URL or data URL
            
        Raises:
            ValueError: If the image data is invalid or cannot be processed
        """
        if data.startswith(('http://', 'https://')):
            # URL - return as is
            return data
        elif data.startswith('data:'):
            # Data URL - validate format
            if ';base64,' in data:
                return data
            else:
                raise ValueError("Invalid data URL format - missing base64 indicator")
        else:
            # Assume raw base64 - try to detect MIME type
            try:
                # Validate and decode base64 once
                decoded = base64.b64decode(data, validate=True)
                
                # Try to detect image type from magic bytes
                header = decoded[:16]  # First few bytes for type detection
                mime_type = None
                
                if header.startswith(b'\xff\xd8\xff'):
                    mime_type = 'image/jpeg'
                elif header.startswith(b'\x89PNG'):
                    mime_type = 'image/png'
                elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
                    mime_type = 'image/gif'
                elif header.startswith(b'RIFF') and b'WEBP' in decoded[:20]:
                    mime_type = 'image/webp'
                else:
                    # Could not detect image type from magic bytes
                    raise ValueError(
                        "Unable to detect image type from base64 data. "
                        "Please provide a data URL with explicit MIME type or use a supported image format "
                        "(JPEG, PNG, GIF, or WebP)"
                    )
                
                return f"data:{mime_type};base64,{data}"
            except Exception as e:
                raise ValueError(f"Invalid base64 image data: {str(e)}")

    def _build_message_content(self, prompt: llm.Prompt) -> Union[str, List[Union[TextContent, ImageContent]]]:
        """Build message content, handling multimodal inputs for vision-capable models.
        
        Constructs the content field of a message based on the model's capabilities
        and the presence of attachments. For vision-capable models with image
        attachments, returns an array format with text and image_url objects.
        For non-vision models or prompts without attachments, returns plain text.
        
        Args:
            prompt: The prompt object potentially containing text and attachments
            
        Returns:
            str or list: Plain text string for text-only content, or list of
                        content objects for multimodal content
        """
        # Check if model supports vision
        supports_vision = self._get_model_capability("supports_vision")
        
        # Check for attachments
        if hasattr(prompt, 'attachments') and prompt.attachments and supports_vision:
            content: List[Union[TextContent, ImageContent]] = [{"type": "text", "text": prompt.prompt}]
            
            for attachment in prompt.attachments:
                if attachment.type == "image":
                    try:
                        # Get image data from attachment
                        if attachment.url:
                            # URL attachment
                            formatted_url = self._validate_image_format(attachment.url)
                        elif attachment.content or attachment.path:
                            # Content or file attachment - convert to base64 data URL
                            base64_data = attachment.base64_content()  # type: ignore[no-untyped-call]
                            # Create data URL with image MIME type
                            mime_type = attachment.resolve_type() or "image/jpeg"  # type: ignore[no-untyped-call]
                            data_url = f"data:{mime_type};base64,{base64_data}"
                            formatted_url = self._validate_image_format(data_url)
                        else:
                            raise ValueError("Attachment has no content, path, or URL")
                        
                        image_content: ImageContent = {
                            "type": "image_url",
                            "image_url": {"url": formatted_url}
                        }
                        content.append(image_content)
                    except ValueError as e:
                        # Log error but continue with other attachments
                        if "pytest" not in sys.modules:
                            console.print(f"[yellow]Warning: Skipping invalid image - {str(e)}[/yellow]")
            
            return content
        
        # Return plain text for non-multimodal or unsupported models
        return prompt.prompt

    def build_messages(self, prompt: llm.Prompt, conversation: Optional[llm.Conversation]) -> List[Message]:
        messages: List[Message] = []

        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})

        if conversation:
            for prev_response in conversation.responses:
                if prev_response.prompt.system:
                    messages.append(
                        {"role": "system", "content": prev_response.prompt.system}
                    )
                messages.append(
                    {"role": "user", "content": self._build_message_content(prev_response.prompt)}
                )
                # Response objects have a text() method
                response_text: str = ""
                if hasattr(prev_response, 'text') and callable(getattr(prev_response, 'text')):
                    response_text = cast(Any, prev_response).text()
                else:
                    response_text = str(prev_response)
                assistant_msg: Message = {"role": "assistant", "content": response_text}
                messages.append(assistant_msg)

        user_content = self._build_message_content(prompt)
        user_msg: Message = {"role": "user", "content": user_content}
        messages.append(user_msg)
        return messages

    def _handle_rate_limit(self, response: httpx.Response, attempt: int) -> None:
        retry_after = response.headers.get("Retry-After")

        if retry_after:
            try:
                wait_time = int(retry_after)
                if attempt < self.MAX_RETRIES - 1:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        task = progress.add_task(
                            f"Rate limit hit. Waiting {wait_time}s as suggested by API...",
                            total=wait_time,
                        )
                        while not progress.finished:
                            time.sleep(0.1)
                            progress.update(task, advance=0.1)
                    return
            except ValueError:
                pass

        if attempt < self.MAX_RETRIES - 1:
            delay = self.BASE_DELAY * (2**attempt)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    f"Rate limit hit. Retrying in {delay}s...", total=delay
                )
                while not progress.finished:
                    time.sleep(0.1)
                    progress.update(task, advance=0.1)
            return

        try:
            error_details = response.json()
            if "error" in error_details:
                error_message = error_details["error"].get("message", "")
                if (
                    "quota exceeded" in error_message.lower()
                    or "insufficient credits" in error_message.lower()
                ):
                    raise QuotaExceededError(
                        "API Quota Exceeded",
                        {
                            "message": "Your x.ai API quota has been exceeded or you have insufficient credits.\n"
                            "Please visit https://x.ai to check your account status."
                        },
                    )
        except (json.JSONDecodeError, AttributeError, KeyError):
            pass

        raise RateLimitError(
            "Rate Limit Exceeded",
            {
                "message": "You've hit the API rate limit. This could mean:\n"
                "1. Too many requests in a short time\n"
                "2. Your account has run out of credits\n\n"
                "Please visit https://x.ai to check your account status\n"
                "or wait a few minutes before trying again."
            },
        )

    def _make_request(self, client: httpx.Client, method: str, url: str, headers: Dict[str, str], json_data: Dict[str, Any], stream: bool = False) -> httpx.Response:
        """Execute HTTP request with retry logic for rate limiting.
        
        Args:
            client: httpx.Client or httpx.AsyncClient instance
            method: HTTP method (e.g., 'POST')
            url: Request URL
            headers: Request headers dict
            json_data: JSON payload for request body
            stream: Whether to stream the response
            
        Returns:
            httpx.Response or stream context manager
            
        Raises:
            RateLimitError: When rate limit is exceeded after all retries
            QuotaExceededError: When API quota is exceeded
            GrokError: For other API errors
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                if stream:
                    return client.stream(
                        method, url, headers=headers, json=json_data, timeout=None
                    )
                else:
                    return client.request(
                        method, url, headers=headers, json=json_data, timeout=None
                    )
            except httpx.HTTPError as e:
                if (
                    hasattr(e, "response")
                    and e.response is not None
                    and e.response.status_code == 429
                ):
                    if self._handle_rate_limit(e.response, attempt):
                        continue
                raise

    def _convert_to_anthropic_messages(self, openai_messages: List[Message]) -> AnthropicRequest:
        """Convert OpenAI-style messages to Anthropic format.
        
        Args:
            openai_messages: List of OpenAI-format messages
            
        Returns:
            Dict with 'messages', 'system' (optional), suitable for Anthropic API
        """
        system_prompts = []
        anthropic_messages = []
        
        for msg in openai_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                # Collect system messages
                if isinstance(content, str):
                    system_prompts.append(content)
                else:
                    # If content is multimodal, extract text parts
                    for part in content:
                        if part["type"] == "text":
                            system_prompts.append(part["text"])
            
            elif role in ["user", "assistant"]:
                # Convert user/assistant messages
                anthropic_role = cast(Literal["user", "assistant"], role)
                anthropic_msg: AnthropicMessage = {"role": anthropic_role, "content": []}
                
                if isinstance(content, str):
                    text_block: AnthropicTextBlock = {
                        "type": "text",
                        "text": content
                    }
                    anthropic_msg["content"] = [text_block]
                else:
                    # Handle multimodal content
                    anthropic_content: List[Union[AnthropicTextBlock, AnthropicImage, AnthropicToolUse]] = []
                    for part in content:
                        if part["type"] == "text":
                            text_block: AnthropicTextBlock = {
                                "type": "text",
                                "text": part["text"]
                            }
                            anthropic_content.append(text_block)
                        elif part["type"] == "image_url":
                            # Convert image format
                            image_url = part["image_url"]["url"]
                            image_data = self._convert_image_to_anthropic(image_url)
                            if image_data:
                                anthropic_content.append(image_data)
                    anthropic_msg["content"] = anthropic_content
                
                # Handle tool calls in assistant messages
                if role == "assistant" and "tool_calls" in msg:
                    # Add tool uses to content array
                    tool_uses = self._convert_tool_calls_to_anthropic(msg["tool_calls"])
                    if isinstance(anthropic_msg["content"], list):
                        anthropic_msg["content"].extend(tool_uses)
                    else:
                        # This shouldn't happen, but just in case
                        text_block: AnthropicTextBlock = {"type": "text", "text": str(anthropic_msg["content"])}
                        content_list: List[Union[AnthropicTextBlock, AnthropicImage, AnthropicToolUse]] = [text_block]
                        content_list.extend(tool_uses)
                        anthropic_msg["content"] = content_list
                
                anthropic_messages.append(anthropic_msg)
        
        result: AnthropicRequest = {"messages": anthropic_messages}
        
        # Combine system prompts if any
        if system_prompts:
            result["system"] = "\n\n".join(system_prompts)
            if len(system_prompts) > 1:
                console.print("[yellow]Warning: Multiple system messages combined into one[/yellow]")
        
        return result
    
    def _convert_image_to_anthropic(self, image_url: str) -> Optional[AnthropicImage]:
        """Convert OpenAI image URL to Anthropic image format."""
        MAX_IMAGE_SIZE = 1024 * 1024  # 1MB limit
        FETCH_TIMEOUT = 10  # 10 seconds timeout
        
        try:
            if image_url.startswith("data:"):
                # Parse data URL
                media_type, base64_data = self._parse_data_url(image_url)
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": base64_data
                    }
                }
            elif image_url.startswith(("http://", "https://")):
                # For HTTP URLs, fetch and convert to base64
                try:
                    import httpx
                    console.print("[dim]Fetching image from URL...[/dim]")
                    with httpx.Client() as client:
                        # Stream the response to check size before downloading
                        with client.stream("GET", image_url, timeout=FETCH_TIMEOUT) as response:
                            response.raise_for_status()
                            
                            # Check content type
                            content_type = response.headers.get("content-type", "image/jpeg")
                            if not content_type.startswith("image/"):
                                console.print(f"[yellow]Warning: URL does not point to an image (content-type: {content_type})[/yellow]")
                                return None
                            
                            # Check content length if available
                            content_length = response.headers.get("content-length")
                            if content_length and int(content_length) > MAX_IMAGE_SIZE:
                                console.print(f"[yellow]Warning: Image too large ({int(content_length)/1024/1024:.1f}MB > 1MB limit)[/yellow]")
                                return None
                            
                            # Download with size limit
                            chunks = []
                            total_size = 0
                            for chunk in response.iter_bytes():
                                total_size += len(chunk)
                                if total_size > MAX_IMAGE_SIZE:
                                    console.print(f"[yellow]Warning: Image download exceeded 1MB limit[/yellow]")
                                    return None
                                chunks.append(chunk)
                            
                            # Convert to base64
                            import base64
                            image_data = base64.b64encode(b"".join(chunks)).decode("utf-8")
                            
                            return {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": content_type,
                                    "data": image_data
                                }
                            }
                except httpx.TimeoutException:
                    console.print(f"[yellow]Warning: Image fetch timed out after {FETCH_TIMEOUT}s[/yellow]")
                    return None
                except httpx.HTTPStatusError as e:
                    console.print(f"[yellow]Warning: Failed to fetch image: HTTP {e.response.status_code}[/yellow]")
                    return None
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to fetch image: {str(e)}[/yellow]")
                    return None
            else:
                # Assume it's already base64
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",  # Default, should be detected
                        "data": image_url
                    }
                }
        except Exception as e:
            console.print(f"[red]Error converting image: {e}[/red]")
            return None
    
    def _parse_data_url(self, data_url: str) -> tuple[str, str]:
        """Parse data URL to extract media type and base64 data."""
        if not data_url.startswith("data:"):
            raise ValueError("Not a data URL")
        
        header, data = data_url[5:].split(",", 1)
        media_type = header.split(";")[0] if ";" in header else header
        return media_type, data
    
    def _convert_tool_calls_to_anthropic(self, tool_calls: List[ToolCall]) -> List[AnthropicToolUse]:
        """Convert OpenAI tool calls to Anthropic tool_use format."""
        anthropic_tools = []
        for call in tool_calls:
            try:
                arguments = json.loads(call["function"]["arguments"])
            except json.JSONDecodeError as e:
                console.print(f"[yellow]Warning: Failed to parse tool arguments: {e}[/yellow]")
                arguments = {}  # Use empty dict as fallback
            
            anthropic_tools.append({
                "id": call.get("id", ""),
                "name": call["function"]["name"],
                "input": arguments
            })
        return anthropic_tools
    
    def _convert_tools_to_anthropic(self, openai_tools: List[ToolDefinition]) -> List[AnthropicToolDefinition]:
        """Convert OpenAI tool definitions to Anthropic format."""
        anthropic_tools = []
        for tool in openai_tools:
            if tool["type"] == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {})
                })
        return anthropic_tools
    
    def _convert_from_anthropic_response(self, anthropic_response: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Anthropic response to OpenAI format."""
        openai_response = {
            "id": anthropic_response.get("id", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": anthropic_response.get("model", self.model_id),
            "choices": [],
            "usage": anthropic_response.get("usage", {})
        }
        
        # Convert content
        content = anthropic_response.get("content", [])
        if isinstance(content, list):
            # Extract text content
            text_parts = [part["text"] for part in content if part["type"] == "text"]
            message_content = "".join(text_parts)
            
            # Extract tool uses
            tool_uses = [part for part in content if part["type"] == "tool_use"]
            tool_calls = None
            if tool_uses:
                tool_calls = []
                for i, tool_use in enumerate(tool_uses):
                    tool_calls.append({
                        "id": tool_use.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": tool_use["name"],
                            "arguments": json.dumps(tool_use.get("input", {}))
                        }
                    })
        else:
            message_content = content
            tool_calls = None
        
        message = {
            "role": "assistant",
            "content": message_content
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        
        openai_response["choices"].append({
            "index": 0,
            "message": message,
            "finish_reason": anthropic_response.get("stop_reason", "stop")
        })
        
        return openai_response
    
    def _convert_anthropic_stream_chunk(self, event_type: str, event_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert Anthropic streaming event to OpenAI format chunk."""
        if event_type == "message_start":
            # Initial message metadata
            return {
                "id": event_data.get("message", {}).get("id", ""),
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": event_data.get("message", {}).get("model", self.model_id),
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }]
            }
        
        elif event_type == "content_block_start":
            # Start of a content block
            block = event_data.get("content_block", {})
            if block.get("type") == "text":
                return {
                    "choices": [{
                        "index": 0,
                        "delta": {"content": ""},
                        "finish_reason": None
                    }]
                }
            elif block.get("type") == "tool_use":
                # Start of tool use
                return {
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": event_data.get("index", 0),
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": ""
                                }
                            }]
                        },
                        "finish_reason": None
                    }]
                }
        
        elif event_type == "content_block_delta":
            # Content delta
            delta = event_data.get("delta", {})
            if delta.get("type") == "text_delta":
                return {
                    "choices": [{
                        "index": 0,
                        "delta": {"content": delta.get("text", "")},
                        "finish_reason": None
                    }]
                }
            elif delta.get("type") == "input_json_delta":
                # Tool input delta
                return {
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": event_data.get("index", 0),
                                "function": {
                                    "arguments": delta.get("partial_json", "")
                                }
                            }]
                        },
                        "finish_reason": None
                    }]
                }
        
        elif event_type == "message_delta":
            # Message metadata updates (like stop_reason)
            delta = event_data.get("delta", {})
            if "stop_reason" in delta:
                return {
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": delta["stop_reason"]
                    }]
                }
        
        elif event_type == "message_stop":
            # Final message
            return {
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
        
        return None
    
    def _parse_anthropic_sse(self, buffer: str) -> tuple[Optional[tuple[str, Dict[str, Any]]], str]:
        """Parse Anthropic SSE format and return (event_type, event_data) tuple and remaining buffer."""
        if "\n\n" not in buffer:
            return None, buffer
            
        message, remaining_buffer = buffer.split("\n\n", 1)
        lines = message.strip().split("\n")
        event_type = None
        event_data = None
        
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
                if data != "[DONE]":
                    try:
                        event_data = json.loads(data)
                    except json.JSONDecodeError:
                        continue
        
        if event_type and event_data:
            return (event_type, event_data), remaining_buffer
        return None, remaining_buffer
    
    def _parse_openai_sse(self, buffer: str) -> tuple[Optional[Dict[str, Any]], str]:
        """Parse OpenAI SSE format and return parsed data and remaining buffer."""
        if "\n\n" not in buffer:
            return None, buffer
            
        message, remaining_buffer = buffer.split("\n\n", 1)
        if message.startswith("data: "):
            data = message[6:]
            if data == "[DONE]":
                return {"done": True}, remaining_buffer
            try:
                return json.loads(data), remaining_buffer
            except json.JSONDecodeError as e:
                console.print(
                    f"[yellow]Warning: Failed to parse JSON in streaming response: {str(e)}. "
                    f"Skipping chunk: {data[:100]}{'...' if len(data) > 100 else ''}[/yellow]"
                )
        return None, remaining_buffer
    
    def _process_stream_delta(self, delta: Dict[str, Any], response: llm.Response) -> Optional[str]:
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

    def _finalize_tool_calls(self, response: llm.Response) -> None:
        """Convert accumulated tool calls to proper llm.ToolCall objects."""
        if hasattr(response, '_tool_calls_accumulator') and getattr(response, '_tool_calls_accumulator', None):
            # Check if this is a real llm.Response or a mock
            if hasattr(response, 'add_tool_call'):
                # Real llm.Response - use the proper method
                tool_calls_accumulator = getattr(response, '_tool_calls_accumulator', [])
                for tool_call_data in tool_calls_accumulator:
                    if tool_call_data.get("function") and tool_call_data["function"].get("name"):
                        try:
                            # Parse the accumulated arguments JSON
                            arguments = json.loads(tool_call_data["function"].get("arguments", "{}"))
                        except json.JSONDecodeError:
                            # If parsing fails, use empty dict
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
                setattr(response, 'tool_calls', getattr(response, '_tool_calls_accumulator', []))
            
            # Clean up the accumulator
            delattr(response, '_tool_calls_accumulator')

    def execute(self, prompt: llm.Prompt, stream: bool, response: llm.Response, conversation: Optional[llm.Conversation], key: Optional[str] = None) -> Iterator[str]:
        key = self.get_key(key)
        messages = self.build_messages(prompt, conversation)
        setattr(response, '_prompt_json', {"messages": messages})

        if not hasattr(prompt, "options") or not isinstance(
            prompt.options, self.Options
        ):
            options = self.Options()
        else:
            options = prompt.options
        
        # Type assertion for PyLance
        assert isinstance(options, self.Options)

        # Determine which endpoint to use
        use_messages = options.use_messages_endpoint
        url = self.MESSAGES_URL if use_messages else self.API_URL
        
        # Check for multimodal content with non-vision models
        has_images = any(
            isinstance(msg.get("content"), list) and 
            any(isinstance(part, dict) and part.get("type") == "image_url" for part in msg.get("content", []))
            for msg in messages
        )
        
        if has_images and not self._get_model_capability("supports_vision"):
            console.print(
                f"[yellow]Warning: Model '{self.model_id}' does not support vision. Image content will be ignored.[/yellow]"
            )
        
        # Build request body based on endpoint
        if use_messages:
            # Convert to Anthropic format
            anthropic_data = self._convert_to_anthropic_messages(messages)
            body = {
                "model": self.model_id,
                "stream": stream,
                "temperature": options.temperature,
            }
            
            # Add messages and system from conversion
            body["messages"] = anthropic_data.get("messages", [])
            if "system" in anthropic_data:
                body["system"] = anthropic_data["system"]
        else:
            # Standard OpenAI format
            body = {
                "model": self.model_id,
                "messages": messages,
                "stream": stream,
                "temperature": options.temperature,
            }

        # Get model info for capability checks
        supports_tools = self._get_model_capability("supports_tools")

        if options.max_completion_tokens is not None:
            # TODO: If max_completion_tokens runs out during reasoning, llm will crash when trying to log to db
            # This happens because Grok 4's automatic reasoning tokens can exceed the limit before generating
            # actual output tokens, causing the response to be empty. Consider implementing a minimum buffer
            # or warning when using low max_completion_tokens values with Grok 4 models.
            
            # Validate max_completion_tokens against model's limit
            model_info = MODEL_INFO.get(self.model_id, {})
            max_output_tokens = model_info.get("max_output_tokens")
            
            if max_output_tokens and options.max_completion_tokens > max_output_tokens:
                console.print(
                    f"[yellow]Warning: max_completion_tokens ({options.max_completion_tokens}) "
                    f"exceeds model's limit ({max_output_tokens}). Clamping to model limit.[/yellow]"
                )
                # Use appropriate key based on endpoint
                if use_messages:
                    body["max_tokens"] = max_output_tokens
                else:
                    body["max_completion_tokens"] = max_output_tokens
            else:
                # Use appropriate key based on endpoint
                if use_messages:
                    body["max_tokens"] = options.max_completion_tokens
                else:
                    body["max_completion_tokens"] = options.max_completion_tokens
        
        # Add function calling parameters if model supports it
        if supports_tools:
            if options.tools is not None:
                if use_messages:
                    # Convert tools to Anthropic format
                    body["tools"] = self._convert_tools_to_anthropic(options.tools)
                else:
                    body["tools"] = options.tools
                
            if options.tool_choice is not None:
                if use_messages:
                    # Convert tool_choice to Anthropic format
                    if options.tool_choice == "auto":
                        body["tool_choice"] = {"type": "auto"}
                    elif options.tool_choice == "none":
                        body["tool_choice"] = {"type": "none"}
                    elif isinstance(options.tool_choice, dict) and "function" in options.tool_choice:
                        # Specific function - Anthropic uses {"type": "tool", "name": "function_name"}
                        body["tool_choice"] = {
                            "type": "tool",
                            "name": options.tool_choice["function"]["name"]
                        }
                else:
                    body["tool_choice"] = options.tool_choice
                
            if options.response_format is not None:
                if not use_messages:  # Anthropic doesn't support response_format in same way
                    body["response_format"] = options.response_format
                else:
                    console.print("[yellow]Warning: response_format is not supported with messages endpoint[/yellow]")
                
        if options.reasoning_effort is not None:
            body["reasoning_effort"] = options.reasoning_effort

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

        try:
            if stream:
                buffer = ""
                with httpx.Client() as client:
                    with self._make_request(
                        client,
                        "POST",
                        url,
                        headers=headers,
                        json_data=body,
                        stream=True,
                    ) as r:
                        r.raise_for_status()
                        for chunk in r.iter_raw():
                            if chunk:
                                buffer += chunk.decode("utf-8")
                                
                                if use_messages:
                                    # Anthropic SSE format parsing
                                    while True:
                                        result, buffer = self._parse_anthropic_sse(buffer)
                                        if result is None:
                                            break
                                        
                                        event_type, event_data = result
                                        # Convert Anthropic event to OpenAI format
                                        openai_chunk = self._convert_anthropic_stream_chunk(event_type, event_data)
                                        if openai_chunk and "choices" in openai_chunk and openai_chunk["choices"]:
                                            choice = openai_chunk["choices"][0]
                                            delta = choice.get("delta", {})
                                            content = self._process_stream_delta(delta, response)
                                            if content:
                                                yield content
                                else:
                                    # OpenAI SSE format parsing
                                    while True:
                                        parsed_data, buffer = self._parse_openai_sse(buffer)
                                        if parsed_data is None:
                                            break
                                        if parsed_data.get("done"):
                                            break
                                        
                                        if "choices" in parsed_data and parsed_data["choices"]:
                                            choice = parsed_data["choices"][0]
                                            delta = choice.get("delta", {})
                                            content = self._process_stream_delta(delta, response)
                                            if content:
                                                yield content
                        # Finalize tool calls after streaming completes
                        self._finalize_tool_calls(response)
            else:
                with httpx.Client() as client:
                    r = self._make_request(
                        client,
                        "POST",
                        url,
                        headers=headers,
                        json_data=body,
                    )
                    r.raise_for_status()
                    response_data = r.json()
                    
                    # Convert response based on endpoint
                    if use_messages:
                        # Convert from Anthropic format to OpenAI format
                        response_data = self._convert_from_anthropic_response(response_data)
                    
                    response.response_json = response_data
                    if "choices" in response_data and response_data["choices"]:
                        choice = response_data["choices"][0]
                        message = choice["message"]
                        
                        # Handle function/tool calls
                        if "tool_calls" in message and message["tool_calls"]:
                            if hasattr(response, 'add_tool_call'):
                                # Real llm.Response - convert to proper llm.ToolCall objects
                                for tool_call in message["tool_calls"]:
                                    if tool_call.get("function") and tool_call["function"].get("name"):
                                        try:
                                            arguments = json.loads(tool_call["function"].get("arguments", "{}"))
                                        except json.JSONDecodeError:
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
                                response.tool_calls = message["tool_calls"]
                            
                            # For now, yield the content if any (might be None for pure tool calls)
                            if message.get("content"):
                                yield message["content"]
                        else:
                            # Regular content response
                            if message.get("content"):
                                yield message["content"]
        except httpx.HTTPError as e:
            if (
                hasattr(e, "response")
                and e.response is not None
                and e.response.status_code == 429
            ):
                try:
                    self._handle_rate_limit(e.response, self.MAX_RETRIES)
                except (RateLimitError, QuotaExceededError) as rate_error:
                    error_panel = Panel.fit(
                        f"[bold red]{rate_error.message}[/]\n\n[white]{rate_error.details}[/]",
                        title="❌ Error",
                        border_style="red",
                    )
                    if "pytest" in sys.modules:
                        raise rate_error
                    rprint(error_panel)
                    sys.exit(1)

            error_body = None
            if hasattr(e, "response") and e.response is not None:
                try:
                    if e.response.is_stream_consumed:
                        error_body = e.response.text
                    else:
                        error_body = e.response.read().decode("utf-8")
                except (AttributeError, UnicodeDecodeError, OSError):
                    error_body = str(e)

            error_message = f"API Error: {str(e)}"
            if error_body:
                try:
                    error_json = json.loads(error_body)
                    if "error" in error_json and "message" in error_json["error"]:
                        error_message = error_json["error"]["message"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

            error_panel = Panel.fit(
                f"[bold red]API Error[/]\n\n[white]{error_message}[/]",
                title="❌ Error",
                border_style="red",
            )
            if "pytest" in sys.modules:
                raise GrokError(error_message)
            rprint(error_panel)
            sys.exit(1)


@llm.hookimpl
def register_commands(cli: Any) -> None:
    @cli.group()
    def grok() -> None:
        "Commands for the Grok model"

    @grok.command()
    def models() -> None:
        "Show available Grok models"
        click.echo("Available models:")
        for model in AVAILABLE_MODELS:
            if model == DEFAULT_MODEL:
                click.echo(f"  {model} (default)")
            else:
                click.echo(f"  {model}")
