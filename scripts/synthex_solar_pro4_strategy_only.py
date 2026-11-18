#!/usr/bin/env python3
"""
SynthEx StrategyGenerator만 Solar Pro 4로 실행해 전략 생성을 시도하고,
결과를 직접 JSON으로 저장하는 스크립트.

목적
- 체크포인트 무결성 문제와 분리해, Solar Pro 4가 실제 전략 생성을 할 수 있는지 확인한다.
- 성공 시 생성된 전략을 저장한다. 실패 시 실패 원인만 기록한다.
- 생성한 적 없는 전략/경로를 임의로 만들지 않는다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
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

try:
    from synthex.agents import AgentContext, StrategyGenerator, StrategyRequest, StrategyResult
    from synthex.llm import LLMRequest, LLMUsage, Message, ThinkingLevel
except Exception as exc:
    print("synthex import failed:", exc)
    sys.exit(2)

from synthex_adapters.solar import SolarClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("synthex_solar_pro4_strategy_only")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _env_api_key() -> str | None:
    for name in ("SOLAR_API_KEY", "OPENCODE_SOLAR_KEY", "UPSTAGE_API_KEY", "OPENCODE_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _log_event(log_path: Path, event: dict[str, Any]) -> None:
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)


def _write_report(output_dir: Path, status: str, result: dict[str, Any] | None, error: Exception | None, llm_calls: list[dict[str, Any]]) -> dict[str, Any]:
    report = {
        "run_type": "synthex-solar-pro4-strategy-only",
        "target": None,
        "target_smiles": None,
        "target_id": None,
        "model": "solar-pro4",
        "provider": "upstage",
        "base_url": "https://api.upstage.ai/v1/chat/completions",
        "output_dir": str(output_dir),
        "status": status,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "llm_calls": llm_calls,
    }
    if result is not None:
        report.update(result)
    if error is not None:
        report["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        }
    (output_dir / "run-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run_strategy_only(
    smiles: str,
    target_id: str | None,
    constraint: str | None,
    strategy_count: int = 3,
    temperature: float = 0.1,
    thinking_level: str | None = None,
    output_base: str | Path = "runs",
) -> dict[str, Any]:
    safe_id = target_id or smiles or "target"
    safe_name = "".join(ch for ch in safe_id if ch.isalnum() or ch in "._-") or "target"
    output_dir = Path(output_base) / f"{safe_name}-strategy-only"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "execution.log"

    _log_event(log_path, {"event": "start", "target_smiles": smiles, "target_id": target_id, "output_dir": str(output_dir)})

    api_key = _env_api_key()
    if not api_key:
        error = RuntimeError("Solar/Upstage API key not found. Checked: SOLAR_API_KEY, OPENCODE_SOLAR_KEY, UPSTAGE_API_KEY, OPENCODE_API_KEY.")
        _log_event(log_path, {"event": "no_key", "error": repr(error)})
        return _write_report(output_dir, "failed_no_key", None, error, [])

    try:
        client = SolarClient(
            model="solar-pro4",
            api_key=api_key,
            base_url="https://api.upstage.ai/v1",
            timeout_seconds=600,
            max_attempts=3,
        )
    except Exception as exc:
        _log_event(log_path, {"event": "client_error", "error": repr(exc), "traceback": traceback.format_exc()})
        return _write_report(output_dir, "failed_client", None, exc, [])

    agent = StrategyGenerator(temperature=temperature, thinking_level=thinking_level)
    context = AgentContext(llm=client, run_id="strategy-only-solar")
    llm_calls: list[dict[str, Any]] = []

    async def generate() -> StrategyResult:
        request = StrategyRequest(
            target=smiles,
            count=strategy_count,
            constraint=constraint,
        )
        return await agent.run(request, context)

    start = time.perf_counter()
    result: StrategyResult | None = None
    error: Exception | None = None
    try:
        result = asyncio.run(generate())
        _log_event(log_path, {"event": "strategy_generated", "strategy_count": len(result.strategies)})
    except Exception as exc:
        error = exc
        _log_event(log_path, {"event": "strategy_failed", "error": repr(exc), "traceback": traceback.format_exc()})
        logger.error("strategy generation failed: %s", exc)

    duration = round(time.perf_counter() - start, 6)

    report_payload = {
        "strategy_count_requested": strategy_count,
        "strategy_count_actual": len(result.strategies) if result is not None else None,
        "duration_seconds": duration,
    }
    if result is not None:
        report_payload["strategies"] = result.strategies
        report_payload["strategy_model_dump"] = result.model_dump(mode="json")
        status = "complete"
        _log_event(log_path, {"event": "report_written", "status": status})
        logger.info("strategies generated: %s", len(result.strategies))
    else:
        status = "failed"
        _log_event(log_path, {"event": "report_written", "status": status})

    report = _write_report(output_dir, status, report_payload if result is not None else None, error, llm_calls)
    logger.info("status: %s", status)
    logger.info("report: %s", output_dir / "run-report.json")
    logger.info("log: %s", log_path)
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="SynthEx 전략 생성만 Solar Pro 4로 실행")
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--target-id", default=None)
    parser.add_argument("--constraint", default=None)
    parser.add_argument("--strategy-count", type=int, default=3)
    parser.add_argument("--thinking-level", default=None, help="minimal/low/medium/high")
    parser.add_argument("--output-base", default="runs")
    args = parser.parse_args()
    report = run_strategy_only(
        smiles=args.smiles,
        target_id=args.target_id,
        constraint=args.constraint,
        strategy_count=args.strategy_count,
        thinking_level=args.thinking_level,
        output_base=args.output_base,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
