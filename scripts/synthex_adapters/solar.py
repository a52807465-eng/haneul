"""
Solar Pro 4용 SynthEx LLM 어댑터(패키지 모듈).

이 모듈은 SynthEx 소스를 수정하지 않고 Solar Pro 4(Upstage)를
SynthEx의 contract(LLMClient)에 연결하는 용도다.

사용 예
-------
from synthex_adapters.solar import SolarClient

client = SolarClient(
    api_key="...",
    model="solar-pro4",
    base_url="https://api.upstage.ai/v1",
)
response = await client.generate(request)

주의
----
- 이 모듈은 SynthEx 패키지에 포함되지 않는다. scripts/와 같은 별도 위치에서
  유지하거나, synthex_adapters 같은 별도 패키지로 배포할 수 있다.
- ToolCall.thought_signature는 현재 SolarAdapter에서 채우지 않는다(None).
- usage는 OpenAI 호환 응답 객체에서 추정 가능한 토큰 값만 채운다.

의존성
------
- openai (OpenAI 호환 클라이언트)
- synthex (LLMClient, LLMRequest, LLMResponse 등 계약)
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import AsyncOpenAI

from synthex.llm import LLMRequest, LLMResponse, LLMUsage, Message, ToolCall

SOLAR_DEFAULT_BASE_URL = "https://api.upstage.ai/v1"
SOLAR_DEFAULT_MODEL = "solar-pro4"


class SolarClient:
    """OpenAI 호환 Chat Completions 엔드포인트용 LLMClient 구현.

    SynthEx의 LLMClient 계약만 구현하므로 GeminiClient와 교체해 쓸 수 있다.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = SOLAR_DEFAULT_MODEL,
        base_url: str = SOLAR_DEFAULT_BASE_URL,
        timeout_seconds: float = 600,
        max_attempts: int = 3,
    ) -> None:
        if api_key is None:
            api_key = os.environ.get("SOLAR_API_KEY") or os.environ.get("OPENCODE_SOLAR_KEY") or os.environ.get("UPSTAGE_API_KEY") or os.environ.get("OPENCODE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Solar API key is required. Set SOLAR_API_KEY or pass api_key= explicitly."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self._api_key = api_key
        # base_url은 /chat/completions를 붙이지 않은 v1까지만 준다.
        # AsyncOpenAI가 내부에서 경로를 붙이므로, /chat/completions를 중복해서 넣으면 404가 난다.
        self._openai_base_url = self.base_url

    async def generate(self, request: LLMRequest) -> LLMResponse:
        import asyncio as _asyncio

        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._openai_base_url,
            timeout=self.timeout_seconds,
        )

        messages = self._to_chat_messages(request)
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": request.temperature,
        }

        if request.response_schema is not None:
            params["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "synthex_response",
                "strict": True,
                "schema": request.response_schema,
            }}

        if request.tools:
            params["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in request.tools
            ]
            params["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await _asyncio.wait_for(
                    client.chat.completions.create(**params),
                    timeout=self.timeout_seconds,
                )
                choice = response.choices[0]
                message_dict = self._message_as_dict(choice.message)
                tool_calls = self._from_tool_calls(message_dict.get("tool_calls") or [])
                content = message_dict.get("content") or ""
                usage = self._from_usage(response.usage)
                return LLMResponse(
                    text=content,
                    tool_calls=tuple(tool_calls),
                    usage=usage,
                    model=getattr(response, "model", self.model),
                )
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    await _asyncio.sleep(5 * attempt)
        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------
    # 내부 변환
    # ------------------------------------------------------------------

    @staticmethod
    def _message_as_dict(message: Any) -> dict[str, Any]:
        """ChatCompletionMessage 또는 dict를 dict 인터페이스로 안전하게 다룬다."""
        if isinstance(message, dict):
            return message
        if message is None:
            return {}
        result: dict[str, Any] = {}
        for key in ("content", "role", "tool_calls", "name", "refusal"):
            val = getattr(message, key, None)
            if val is not None:
                result[key] = val
        return result

    @staticmethod
    def _to_chat_messages(request: LLMRequest) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = []
        if request.system:
            parts.append({"role": "system", "content": request.system})
        for message in request.messages:
            parts.append(SolarClient._from_message(message))
        return parts

    @staticmethod
    def _from_message(message: Message) -> dict[str, Any]:
        if message.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": message.tool_call_id or "",
                "name": message.name or "",
                "content": message.content,
            }
        item: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.name:
            item["name"] = message.name
        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]
        return item

    @staticmethod
    def _from_tool_calls(raw: list[Any]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            func = item.get("function") or {}
            args_raw = func.get("arguments") or "{}"
            try:
                arguments = json.loads(args_raw)
            except (TypeError, ValueError):
                arguments = {}
            calls.append(
                ToolCall(
                    id=str(item.get("id") or ""),
                    name=str(func.get("name") or ""),
                    arguments=arguments,
                    thought_signature=None,
                )
            )
        return calls

    @staticmethod
    def _from_usage(raw: Any) -> LLMUsage | None:
        if not raw:
            return None
        try:
            return LLMUsage(
                input_tokens=int(getattr(raw, "prompt_tokens", None) or 0),
                cached_input_tokens=None,
                output_tokens=int(getattr(raw, "completion_tokens", None) or 0),
                thought_tokens=None,
                total_tokens=int(getattr(raw, "total_tokens", None) or 0),
            )
        except (TypeError, ValueError):
            return None
