from __future__ import annotations

from google.genai import types

from synthex.integrations.gemini import GeminiClient
from synthex.llm import Message, ToolCall


def test_assistant_function_call_preserves_gemini_thought_signature():
    signature = b"opaque-gemini-thought-signature"
    message = Message(
        role="assistant",
        tool_calls=(
            ToolCall(
                id="call-1",
                name="read_plan",
                arguments={},
                thought_signature=signature,
            ),
        ),
    )

    content = GeminiClient._message_content(message)

    assert content.role == "model"
    assert content.parts is not None
    assert content.parts[0].function_call == types.FunctionCall(name="read_plan", args={})
    assert content.parts[0].thought_signature == signature
