"""Small provider-independent value types. No agent framework is involved."""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable
from uuid import uuid4


def new_id() -> str:
    return uuid4().hex


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Completion:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    finish_reason: str = "stop"
    # Exact response blocks/items needed to replay provider-specific reasoning.
    provider_payload: Any = None
    reasoning: str | None = None


DeltaCallback = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class ProviderConfig:
    protocol: str = "openai_chat"
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    context_window: int = 32768
    max_output_tokens: int = 4096
    timeout: float = 60.0
    max_retries: int = 2
    strict_schema: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
