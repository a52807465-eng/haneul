"""Agent protocol and injected runtime context."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from synthex.llm import LLMClient

InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)


@dataclass(frozen=True)
class AgentContext:
    llm: LLMClient
    run_id: str = "offline"
    event_sink: Callable[[dict[str, Any]], None] | None = None

    def emit(self, event: dict[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(event)


class Agent(Protocol, Generic[InputT, OutputT]):
    name: str
    prompt_version: str

    async def run(self, request: InputT, context: AgentContext) -> OutputT: ...


__all__ = ["Agent", "AgentContext"]
