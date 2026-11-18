from __future__ import annotations

import asyncio
import os

import pytest

from synthex.integrations.gemini import GeminiClient
from synthex.llm import LLMRequest, Message


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("SYNTHEX_RUN_LIVE") != "1",
    reason="set SYNTHEX_RUN_LIVE=1 to make a credentialed Gemini call",
)
def test_gemini_transport_smoke():
    client = GeminiClient(timeout_seconds=60, max_attempts=1)
    response = asyncio.run(
        client.generate(
            LLMRequest(
                system="Reply concisely.",
                messages=(Message(role="user", content="Return the word OK."),),
                temperature=0.1,
            )
        )
    )
    assert response.text.strip()
