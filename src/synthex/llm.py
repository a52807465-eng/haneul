"""Provider-neutral language-model contracts."""

from __future__ import annotations

from collections import deque
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

ThinkingLevel = Literal["minimal", "low", "medium", "high"]


class ToolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Provider-owned opaque continuation data. Gemini 3.x requires this
    # signature when a function call is replayed in the next model turn.
    thought_signature: bytes | None = None


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant", "tool"]
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


class LLMUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    thought_tokens: int | None = None
    total_tokens: int | None = None


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    system: str
    messages: tuple[Message, ...]
    temperature: float = Field(ge=0, le=2)
    thinking_level: ThinkingLevel | None = None
    tools: tuple[ToolSpec, ...] = ()
    response_schema: dict[str, Any] | None = None


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: LLMUsage | None = None
    model: str | None = None


class LLMClient(Protocol):
    async def generate(self, request: LLMRequest) -> LLMResponse: ...


class ReplayLLMClient:
    """Deterministic client for offline tests and sanitized replay fixtures."""

    def __init__(self, responses: list[LLMResponse | dict[str, Any]]) -> None:
        self._responses = deque(
            response if isinstance(response, LLMResponse) else LLMResponse.model_validate(response)
            for response in responses
        )
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self._responses:
            raise RuntimeError("ReplayLLMClient fixture has no response left")
        return self._responses.popleft()


__all__ = [
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "LLMUsage",
    "Message",
    "ReplayLLMClient",
    "ThinkingLevel",
    "ToolCall",
    "ToolSpec",
]
