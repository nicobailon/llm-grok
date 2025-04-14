import click
import httpx
import json
import llm
import time
import sys
from pydantic import Field
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

AVAILABLE_MODELS = [
    "grok-3-latest",
    "grok-3-mini-fast-latest",
    "grok-3-mini-latest",
    "grok-3-fast-latest",
    "grok-2-latest",
    "grok-2-vision-latest"
]
DEFAULT_MODEL = "grok-3-latest"

@llm.hookimpl
def register_models(register):
    for model_id in AVAILABLE_MODELS:
        register(Grok(model_id))

class GrokError(Exception):
    """Base exception for Grok API errors"""
    def __init__(self, message, details=None):
        self.message = message
        self.details = details
        super().__init__(message)

class RateLimitError(GrokError):
    """Exception for rate limit errors"""
    pass

class QuotaExceededError(GrokError):
    """Exception for quota exceeded errors"""
    pass

class Grok(llm.Model):
    can_stream = True
    needs_key = "grok"
    key_env_var = "XAI_API_KEY"
    MAX_RETRIES = 3
    BASE_DELAY = 1  # Base delay in seconds

    class Options(llm.Options):
        temperature: Optional[float] = Field(
            description=(
                "Determines the sampling temperature. Higher values like 0.8 increase randomness, "
                "while lower values like 0.2 make the output more focused and deterministic."
            ),
            ge=0,
            le=1,
            default=0.0,
        )
        max_tokens: Optional[int] = Field(
            description="The maximum number of tokens to generate in the completion.",
            ge=0,
            default=None,
        )
    def __init__(self, model_id):
        self.model_id = model_id

    def build_messages(self, prompt, conversation):
        messages = []
        
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        else:
            messages.append({
                "role": "system", 
                "content": "You are Grok, a chatbot inspired by the Hitchhikers Guide to the Galaxy."
            })
        
        if conversation:
            for prev_response in conversation.responses:
                if prev_response.prompt.system:
                    messages.append(
                        {"role": "system", "content": prev_response.prompt.system}
                    )
                messages.append(
                    {"role": "user", "content": prev_response.prompt.prompt}
                )
                messages.append(
                    {"role": "assistant", "content": prev_response.text()}
                )

        messages.append({"role": "user", "content": prompt.prompt})
        return messages

    def _handle_rate_limit(self, response, attempt):
        retry_after = response.headers.get('Retry-After')
        
        if retry_after:
            try:
                wait_time = int(retry_after)
                if attempt < self.MAX_RETRIES - 1:
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console,
                    ) as progress:
                        task = progress.add_task(f"Rate limit hit. Waiting {wait_time}s as suggested by API...", total=wait_time)
                        while not progress.finished:
                            time.sleep(1)
                            progress.update(task, advance=1)
                    return True
            except ValueError:
                pass
        
        if attempt < self.MAX_RETRIES - 1:
            delay = self.BASE_DELAY * (2 ** attempt)
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(f"Rate limit hit. Retrying in {delay}s...", total=delay)
                while not progress.finished:
                    time.sleep(1)
                    progress.update(task, advance=1)
            return True
        
        try:
            error_details = response.json()
            if 'error' in error_details:
                error_message = error_details['error'].get('message', '')
                if 'quota exceeded' in error_message.lower() or 'insufficient credits' in error_message.lower():
                    raise QuotaExceededError(
                        "API Quota Exceeded",
                        "Your x.ai API quota has been exceeded or you have insufficient credits.\n"
                        "Please visit https://x.ai to check your account status."
                    )
        except:
            pass
        
        raise RateLimitError(
            "Rate Limit Exceeded",
            "You've hit the API rate limit. This could mean:\n"
            "1. Too many requests in a short time\n"
            "2. Your account has run out of credits\n\n"
            "Please visit https://x.ai to check your account status\n"
            "or wait a few minutes before trying again."
        )

    def _make_request(self, client, method, url, headers, json_data, stream=False):
        for attempt in range(self.MAX_RETRIES):
            try:
                if stream:
                    return client.stream(method, url, headers=headers, json=json_data, timeout=None)
                else:
                    return client.request(method, url, headers=headers, json=json_data, timeout=None)
            except httpx.HTTPError as e:
                if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                    if self._handle_rate_limit(e.response, attempt):
                        continue
                raise

    def execute(self, prompt, stream, response, conversation):
        key = self.get_key()
        messages = self.build_messages(prompt, conversation)
        response._prompt_json = {"messages": messages}

        if not hasattr(prompt, 'options') or not isinstance(prompt.options, self.Options):
            options = self.Options()
        else:
            options = prompt.options

        body = {
            "model": self.model_id,
            "messages": messages,
            "stream": stream,
            "temperature": options.temperature,
        }

        if options.max_tokens is not None:
            body["max_tokens"] = options.max_tokens

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

        try:
            if stream:
                buffer = ""
                with httpx.Client() as client:
                    with self._make_request(
                        client, "POST", "https://api.x.ai/v1/chat/completions",
                        headers=headers, json_data=body, stream=True
                    ) as r:
                        r.raise_for_status()
                        for chunk in r.iter_raw():
                            if chunk:
                                buffer += chunk.decode('utf-8')
                                while '\n\n' in buffer:
                                    message, buffer = buffer.split('\n\n', 1)
                                    if message.startswith('data: '):
                                        data = message[6:]
                                        if data == '[DONE]':
                                            break
                                        try:
                                            parsed = json.loads(data)
                                            if "choices" in parsed and parsed["choices"]:
                                                delta = parsed["choices"][0].get("delta", {})
                                                if "content" in delta:
                                                    content = delta["content"]
                                                    if content:
                                                        yield content
                                        except json.JSONDecodeError:
                                            continue
            else:
                with httpx.Client() as client:
                    r = self._make_request(
                        client, "POST", "https://api.x.ai/v1/chat/completions",
                        headers=headers, json_data=body
                    )
                    r.raise_for_status()
                    response_data = r.json()
                    response.response_json = response_data
                    if "choices" in response_data and response_data["choices"]:
                        yield response_data["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                try:
                    self._handle_rate_limit(e.response, self.MAX_RETRIES)
                except (RateLimitError, QuotaExceededError) as rate_error:
                    error_panel = Panel.fit(
                        f"[bold red]{rate_error.message}[/]\n\n[white]{rate_error.details}[/]",
                        title="❌ Error",
                        border_style="red"
                    )
                    if 'pytest' in sys.modules:
                        raise rate_error
                    rprint(error_panel)
                    sys.exit(1)
            
            error_body = None
            if hasattr(e, 'response') and e.response is not None:
                try:
                    if e.response.is_stream_consumed:
                        error_body = e.response.text
                    else:
                        error_body = e.response.read().decode('utf-8')
                except:
                    error_body = str(e)
            
            error_message = f"API Error: {str(e)}"
            if error_body:
                try:
                    error_json = json.loads(error_body)
                    if 'error' in error_json and 'message' in error_json['error']:
                        error_message = error_json['error']['message']
                except:
                    pass
            
            error_panel = Panel.fit(
                f"[bold red]API Error[/]\n\n[white]{error_message}[/]",
                title="❌ Error",
                border_style="red"
            )
            if 'pytest' in sys.modules:
                raise GrokError(error_message)
            rprint(error_panel)
            sys.exit(1)

@llm.hookimpl
def register_commands(cli):
    @cli.group()
    def grok():
        "Commands for the Grok model"

    @grok.command()
    def models():
        "Show available Grok models"
        click.echo("Available models:")
        for model in AVAILABLE_MODELS:
            if model == DEFAULT_MODEL:
                click.echo(f"  {model} (default)")
            else:
                click.echo(f"  {model}")
