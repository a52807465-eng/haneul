"""Stable parsing helpers for model responses."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        obj = re.search(r"\{.*\}", candidate, re.DOTALL)
        if obj:
            return json.loads(obj.group(0))
        raise ValueError(f"No valid JSON object in model response: {text[:300]}") from None
