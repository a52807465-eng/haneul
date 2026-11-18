"""
Solar Pro 4 / Upstage Chat Completions 계약 검증용 경량 프로브 (revised).

목표
- scripts/synthex_adapters/solar.py 의 SolarClient.generate 가 실제로
  /v1/chat/completions 과 대화할 수 있는지 확인한다.
- SynthEx가 의존하는 세 가지 패턴(최소 요청, structured output, tool call)이
  Upstage 엔드포인트에서 어떻게 동작하는지 기록한다.

수행 방식
- SolarClient.generate(LLMRequest) 만 사용한다.
- API 키는 env에서 읽는다: SOLAR_API_KEY 또는 OPENCODE_SOLAR_KEY.
- 키가 없으면 실제 호출 없이 "failed_no_key"로 기록한다.

출력
- runs/aspirin-solar-pro4/probe-report.json
- runs/aspirin-solar-pro4/execution.log
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNTHX_SRC = REPO_ROOT / "src"
SCRIPTS_SRC = REPO_ROOT / "scripts"
for p in (SYNTHX_SRC, SCRIPTS_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from synthex.llm import LLMRequest, Message, ToolSpec
from synthex_adapters.solar import SolarClient


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _env_api_key() -> str | None:
    for name in ("SOLAR_API_KEY", "OPENCODE_SOLAR_KEY", "UPSTAGE_API_KEY", "OPENCODE_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _write_report(report_dir: Path, status: str, payload: dict[str, Any]) -> None:
    report = {
        "probe": "solar_pro4_chat_completions_contract_probe_v2",
        "model": "solar-pro4",
        "base_url": "https://api.upstage.ai/v1/chat/completions",
        "status": status,
        "started_at": _now_iso(),
        "finished_at": _now_iso(),
        **payload,
    }
    (report_dir / "probe-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _log_event(log_path: Path, event: dict[str, Any]) -> None:
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)


def _summarize_response(resp: Any, include_text_head: bool = True) -> dict[str, Any]:
    text = getattr(resp, "text", None) or ""
    tool_calls = getattr(resp, "tool_calls", None) or ()
    usage = None if resp.usage is None else resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else None
    return {
        "response_model": getattr(resp, "model", None),
        "text_len": len(text),
        "text_head": text[:800] if include_text_head else None,
        "tool_call_count": len(tool_calls) if tool_calls else 0,
        "usage": usage,
    }


async def probe_minimal(client: SolarClient, log_path: Path) -> dict[str, Any]:
    request = LLMRequest(
        system="You are a concise assistant.",
        messages=(Message(role="user", content="Translate 'aspirin' into one sentence of Korean."),),
        temperature=0.3,
    )
    resp = await client.generate(request)
    return {"name": "minimal", **_summarize_response(resp)}


async def probe_structured(client: SolarClient, log_path: Path) -> dict[str, Any]:
    from synthex.agents.models import StrategyResult
    schema = StrategyResult.model_json_schema()
    request = LLMRequest(
        system="You are a synthesis planning assistant.",
        messages=(Message(role="user", content="TARGET MOLECULE SMILES: CC(=O)Oc1ccccc1C(=O)O"),),
        temperature=0.3,
        response_schema=schema,
    )
    resp = await client.generate(request)
    return {"name": "structured", "schema_used": "StrategyResult", **_summarize_response(resp)}


async def probe_tool(client: SolarClient, log_path: Path) -> dict[str, Any]:
    tools = (
        ToolSpec(
            name="get_target_info",
            description="Return basic info about a target molecule.",
            parameters={
                "type": "object",
                "properties": {"smiles": {"type": "string"}},
                "required": ["smiles"],
            },
        ),
    )
    request = LLMRequest(
        system="You are a helper that may call tools.",
        messages=(Message(role="user", content="SMILES: CC(=O)Oc1ccccc1C(=O)O"),),
        temperature=0.3,
        tools=tools,
    )
    resp = await client.generate(request)
    return {
        "name": "tool",
        "tool_calls": [
            {
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            }
            for tc in getattr(resp, "tool_calls", tuple())
        ],
        **_summarize_response(resp),
    }


async def main() -> None:
    project_root = REPO_ROOT
    report_dir = project_root / "runs" / "aspirin-solar-pro4"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / "execution.log"

    _log_event(log_path, {"event": "probe_v2_start", "report_dir": str(report_dir)})

    api_key = _env_api_key()
    if not api_key:
        _log_event(log_path, {"event": "probe_v2_no_key", "checked": ["SOLAR_API_KEY", "OPENCODE_SOLAR_KEY", "UPSTAGE_API_KEY", "OPENCODE_API_KEY"]})
        _write_report(
            report_dir,
            "failed_no_key",
            {
                "error": "No Solar/Upstage API key found in environment.",
                "checked_env": [k for k in ("SOLAR_API_KEY", "OPENCODE_SOLAR_KEY", "UPSTAGE_API_KEY", "OPENCODE_API_KEY") if os.environ.get(k) is not None],
            },
        )
        print("No API key. Probe finished without a real call.")
        print("Report:", report_dir / "probe-report.json")
        return

    try:
        client = SolarClient(
            model="solar-pro4",
            api_key=api_key,
            timeout_seconds=120,
            max_attempts=2,
        )
    except Exception as exc:
        _log_event(log_path, {"event": "probe_v2_client_init_error", "error": repr(exc), "traceback": traceback.format_exc()})
        _write_report(report_dir, "failed_client_init", {"error": repr(exc), "traceback": traceback.format_exc()})
        print("SolarClient init failed:", exc)
        return

    results: dict[str, Any] = {}
    for name, coro in [("minimal", probe_minimal), ("structured", probe_structured), ("tool", probe_tool)]:
        try:
            results[name] = await coro(client, log_path)
            _log_event(log_path, {"event": f"probe_v2_{name}_ok"})
        except Exception as exc:
            results[f"{name}_error"] = {"error": repr(exc), "traceback": traceback.format_exc()}
            _log_event(log_path, {"event": f"probe_v2_{name}_error", "error": repr(exc)})

    ok_any = any(isinstance(v, dict) and "text_len" in v for v in results.values())
    status = "ok" if ok_any else "failed"
    _write_report(report_dir, status, {"probes": results, "api_key_set": bool(api_key)})
    _log_event(log_path, {"event": "probe_v2_finish", "status": status})

    print("Probe v2 finished:", status)
    print("Report:", report_dir / "probe-report.json")
    print("Log:", log_path)


if __name__ == "__main__":
    asyncio.run(main())
