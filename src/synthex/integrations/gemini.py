"""Gemini implementation of the provider-neutral LLM contract."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from google import genai
from google.genai import types

from synthex.llm import LLMRequest, LLMResponse, LLMUsage, Message, ToolCall


class GeminiClient:
    """Async Gemini adapter with bounded retries and no implicit file logging."""

    def __init__(
        self,
        model: str = "gemini-3.1-pro-preview",
        *,
        api_key: str | None = None,
        use_vertex_ai: bool = False,
        timeout_seconds: float = 600,
        max_attempts: int = 3,
        thinking_level: str | None = None,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.thinking_level = thinking_level
        if use_vertex_ai:
            self._client = genai.Client(
                vertexai=True,
                project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
            )
        else:
            resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not resolved_key:
                raise RuntimeError("GEMINI_API_KEY environment variable is not set")
            self._client = genai.Client(api_key=resolved_key)

    @staticmethod
    def _message_content(message: Message) -> types.Content:
        if message.role == "tool":
            return types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name=message.name or "tool",
                        response={"result": message.content},
                    )
                ],
            )
        parts: list[types.Part] = []
        if message.content:
            parts.append(types.Part.from_text(text=message.content))
        for call in message.tool_calls:
            part = types.Part.from_function_call(name=call.name, args=call.arguments)
            part.thought_signature = call.thought_signature
            parts.append(part)
        return types.Content(
            role="model" if message.role == "assistant" else "user",
            parts=parts,
        )

    @staticmethod
    def _tools(request: LLMRequest) -> list[types.Tool] | None:
        if not request.tools:
            return None
        declarations = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=tool.parameters,
            )
            for tool in request.tools
        ]
        return [types.Tool(function_declarations=declarations)]

    async def generate(self, request: LLMRequest) -> LLMResponse:
        config_kwargs: dict[str, Any] = {
            "system_instruction": request.system,
            "temperature": request.temperature,
            "tools": self._tools(request),
        }
        if request.response_schema is not None:
            config_kwargs.update(
                response_mime_type="application/json",
                response_json_schema=request.response_schema,
            )
        thinking_level = request.thinking_level or self.thinking_level
        if thinking_level is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level,
                include_thoughts=False,
            )
        config = types.GenerateContentConfig(**config_kwargs)
        contents = [self._message_content(message) for message in request.messages]

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config,
                    ),
                    timeout=self.timeout_seconds,
                )
                content = response.candidates[0].content if response.candidates else None
                text_parts: list[str] = []
                calls: list[ToolCall] = []
                for index, part in enumerate(getattr(content, "parts", None) or []):
                    if getattr(part, "thought", False):
                        continue
                    if getattr(part, "text", None):
                        text_parts.append(part.text)
                    function_call = getattr(part, "function_call", None)
                    if function_call:
                        calls.append(
                            ToolCall(
                                id=getattr(function_call, "id", None) or f"gemini-call-{index}",
                                name=function_call.name or "",
                                arguments=dict(function_call.args or {}),
                                thought_signature=getattr(part, "thought_signature", None),
                            )
                        )
                metadata = getattr(response, "usage_metadata", None)
                usage = None
                if metadata is not None:
                    usage = LLMUsage(
                        input_tokens=getattr(metadata, "prompt_token_count", None),
                        cached_input_tokens=getattr(metadata, "cached_content_token_count", None),
                        output_tokens=getattr(metadata, "candidates_token_count", None),
                        thought_tokens=getattr(metadata, "thoughts_token_count", None),
                        total_tokens=getattr(metadata, "total_token_count", None),
                    )
                return LLMResponse(
                    text="".join(text_parts),
                    tool_calls=tuple(calls),
                    usage=usage,
                    model=self.model,
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    await asyncio.sleep(5 * attempt)
        assert last_error is not None
        raise last_error


__all__ = ["GeminiClient"]
