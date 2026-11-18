#!/usr/bin/env python3
"""
SynthEx + Solar Pro 4 실행 스크립트 (전략 → 경로 2단계 구동 + 보고서 저장)

동작 방식
- SynthEx 배포 소스를 수정하지 않는다.
- scripts/synthex_adapters/solar.py 의 SolarClient를 LLMClient로 사용한다.
- Pipeline을 stage 단위로 순차적으로 실행한다.
  1) strategies 단계 먼저 실행
  2) 성공 시 routes 단계 실행
  3) 실패 시 즉시 중단하고 실패 단계/오류를 보고서에 기록
- 각 실행 전 출력 디렉터리는 깨끗하게 시작한다. 기존 체크포인트가 남아 있으면 제거 후 새로 시작한다.
- 실행 기록(execution.log)과 최종 보고서(run-report.json)를 runs/<target_id>/ 아래에 저장한다.

주의
- API 키는 env에서만 읽는다. 하드코딩하지 않는다.
- generate되지 않은 전략이나 경로를 임의로 만들지 않는다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
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
    from synthex.agents import (
        AgentContext,
        Analyst,
        AnalystRequest,
        Critic,
        CriticRequest,
        Editor,
        EditorRequest,
        RouteBuilder,
        RouteBuildRequest,
        StrategyGenerator,
        StrategyRequest,
        StrategyResult,
    )
    from synthex.artifacts import ArtifactStore, RunManifest, redact_sensitive_text
    from synthex.config import ManuscriptConfig
    from synthex.llm import LLMClient, LLMRequest, LLMResponse, LLMUsage, Message, ThinkingLevel, ToolCall, ToolSpec
    from synthex.pipeline import PipelineComponents, SynthexPipeline, STAGES, STAGE_DEPENDENCIES
    from synthex.routejson import Route
except Exception as exc:
    print("synthex import failed:", exc)
    sys.exit(2)

from synthex_adapters.solar import SolarClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("synthex_solar_pro4_run_stage")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_name(value: str) -> str:
    import re
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "target"


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


class _RecordingLLM:
    """SolarClient 래퍼: 실제 LLM 호출 결과만 보고서에 기록한다."""

    def __init__(self, client: SolarClient) -> None:
        self.client = client
        self.calls: list[dict[str, Any]] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        started = time.perf_counter()
        try:
            response = await self.client.generate(request)
        except Exception as exc:
            self.calls.append(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": redact_sensitive_text(str(exc)),
                    "duration_seconds": round(time.perf_counter() - started, 6),
                }
            )
            raise
        usage = response.usage
        self.calls.append(
            {
                "status": "ok",
                "model": response.model or self.client.model,
                "text_len": len(response.text),
                "tool_call_count": len(response.tool_calls),
                "usage": (usage.model_dump(mode="json") if usage is not None else None),
                "duration_seconds": round(time.perf_counter() - started, 6),
            }
        )
        return response


def _write_report(
    output_dir: Path,
    status: str,
    manifest: RunManifest | None,
    failed_stage: str | None,
    error: Exception | None,
    llm_calls: list[dict[str, Any]],
    stage_results: list[dict[str, Any]],
    seed_strategies: list[str] | None = None,
) -> dict[str, Any]:
    report = {
        "run_type": "synthex-solar-pro4-staged",
        "target": None,
        "target_smiles": None,
        "target_id": None,
        "model": "solar-pro4",
        "provider": "upstage",
        "base_url": "https://api.upstage.ai/v1/chat/completions",
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "manifest.json") if manifest is not None else None,
        "completed_stages": list(manifest.completed_stages) if manifest is not None else [],
        "status": status,
        "failed_stage": failed_stage,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "llm_calls": llm_calls,
        "stage_results": stage_results,
        "seed_strategies": seed_strategies,
        "stage_status": {
            item["name"]: item["result"].get("status") for item in stage_results
        },
    }
    if manifest is not None:
        report["not_started_stages"] = [
            stage
            for stage in STAGES
            if stage not in manifest.completed_stages and stage != failed_stage
        ]
        if "strategies" in manifest.artifacts:
            try:
                payload = ArtifactStore(output_dir).read_checkpoint(manifest.artifacts["strategies"])
                report["strategy_origin"] = payload.get("origin")
                report["generated_strategies"] = payload.get("strategies")
            except Exception as exc:
                report["generated_strategies_read_error"] = redact_sensitive_text(str(exc))
    if error is not None:
        report["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        }
    if manifest is not None:
        report["target"] = manifest.target
        report["target_smiles"] = manifest.target
        report["target_id"] = manifest.target_id
        report["llm_usage"] = dict(manifest.llm_usage)
        report["stage_durations_seconds"] = dict(manifest.stage_durations_seconds)
    (output_dir / "run-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def _clean_run_dir(output_dir: Path) -> None:
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    output_dir.mkdir(parents=True, exist_ok=True)


def _configure_config(
    strategy_count: int = 3,
    route_step_limit: int = 25,
    route_proposal_attempts: int = 2,
    route_temperature: float = 0.3,
    edit_iteration_limit: int = 6,
    thinking_level: str | None = "low",
    use_completion: bool = False,
) -> ManuscriptConfig:
    return ManuscriptConfig(
        format_version="synthex-config-1.0",
        profile="manuscript",
        llm={
            "provider": "gemini",
            "model": "solar-pro4",
            "use_vertex_ai": False,
            "timeout_seconds": 600,
            "max_attempts": 3,
            "thinking_level": thinking_level if thinking_level is not None else "low",
        },
        agents={
            "strategy_count": strategy_count,
            "route_step_limit": route_step_limit,
            "route_proposal_attempts": route_proposal_attempts,
            "edit_iteration_limit": edit_iteration_limit,
            "editor_turn_limit": 80,
            "strategy_temperature": 0.1,
            "route_temperature": route_temperature,
            "critic_temperature": 0.1,
            "editor_temperature": 0.3,
            "analyst_temperature": 0.3,
            "strategy_thinking_level": None,
            "route_thinking_level": thinking_level if thinking_level is not None else "low",
            "critic_thinking_level": thinking_level if thinking_level is not None else "low",
            "editor_thinking_level": thinking_level if thinking_level is not None else "low",
            "analyst_thinking_level": None,
        },
        completion={
            "enabled": use_completion,
            "aizynthfinder_config": None,
            "stocks": None,
            "expansion_policies": None,
            "filter_policies": None,
            "budget": {"max_transforms": 6, "iteration_limit": 500, "time_limit": 1200, "return_first": True},
        },
        external_asset_manifest=None,
    )


def _make_pipeline(config: ManuscriptConfig, api_key: str) -> tuple[SynthexPipeline, _RecordingLLM]:
    llm = _RecordingLLM(SolarClient(
        model=config.llm.model,
        api_key=api_key,
        base_url="https://api.upstage.ai/v1",
        timeout_seconds=config.llm.timeout_seconds,
        max_attempts=config.llm.max_attempts,
    ))
    components = PipelineComponents(
        llm=llm,
        strategy_generator=StrategyGenerator(
            temperature=config.agents.strategy_temperature,
            thinking_level=config.agents.strategy_thinking_level,
        ),
        route_builder=RouteBuilder(
            temperature=config.agents.route_temperature,
            thinking_level=config.agents.route_thinking_level,
        ),
        critic=Critic(
            temperature=config.agents.critic_temperature,
            thinking_level=config.agents.critic_thinking_level,
        ),
        editor=Editor(
            temperature=config.agents.editor_temperature,
            thinking_level=config.agents.editor_thinking_level,
        ),
        analyst=Analyst(
            temperature=config.agents.analyst_temperature,
            thinking_level=config.agents.analyst_thinking_level,
        ),
    )
    return SynthexPipeline(config, components), llm


def _summarize_checkpoint(store: ArtifactStore, stage: str, manifest: RunManifest) -> dict[str, Any]:
    """체크포인트 읽고 핵심만 요약."""
    reference = manifest.artifacts.get(stage)
    if reference is None:
        return {"stage": stage, "present": False}
    try:
        payload = store.read_checkpoint(reference)
    except Exception as exc:
        return {"stage": stage, "present": True, "read_error": repr(exc)}
    if isinstance(payload, list):
        item_count = len(payload)
        ok_items = sum(1 for item in payload if item.get("status") not in {None, "ok"} or item.get("status") == "ok")
        first_ok = next((item for item in payload if item.get("status") == "ok"), None)
        summary = {
            "stage": stage,
            "present": True,
            "item_count": item_count,
            "ok_item_count": ok_items,
            "first_ok_strategy": first_ok.get("strategy") if first_ok else None,
            "first_ok_route_present": "route" in first_ok if first_ok else None,
        }
        if first_ok and first_ok.get("route"):
            try:
                route = Route.model_validate(first_ok["route"])
                summary["first_ok_route_target"] = route.target
                summary["first_ok_route_step_count"] = len(route.steps)
                summary["first_ok_route_steps"] = [
                    {"id": step.id, "description": step.description, "operation_count": len(step.operations)}
                    for step in route.steps
                ]
            except Exception as exc:
                summary["first_ok_route_parse_error"] = repr(exc)
        return summary
    return {
        "stage": stage,
        "present": True,
        "payload_type": type(payload).__name__,
        "payload_keys": list(payload.keys()) if isinstance(payload, dict) else None,
    }


def _run_stage(
    pipeline: SynthexPipeline,
    output_dir: Path,
    log_path: Path,
    target_smiles: str,
    target_id: str | None,
    constraint: str | None,
    stages: tuple[str, ...],
    stage_started_at: float,
    resume: bool = False,
    seed_strategies: list[str] | None = None,
) -> dict[str, Any]:
    """하나의 stage 묶음을 실행하고 결과를 요약한다."""
    try:
        manifest = asyncio.run(
            pipeline.run(
                target=target_smiles,
                output=str(output_dir),
                target_id=target_id,
                constraint=constraint,
                stages=stages,
                resume=resume,
                seed_strategies=seed_strategies,
            )
        )
    except Exception as exc:
        return {
            "stages_requested": list(stages),
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "stage_started_at": _now_iso(),
            "stage_duration_seconds": round(time.perf_counter() - stage_started_at, 6),
        }

    stage_summaries = []
    for stage in stages:
        summary = _summarize_checkpoint(ArtifactStore(output_dir), stage, manifest)
        summary["duration_seconds"] = manifest.stage_durations_seconds.get(stage)
        stage_summaries.append(summary)

    return {
        "stages_requested": list(stages),
        "status": manifest.status,
        "completed_stages": list(manifest.completed_stages),
        "run_id": manifest.run_id,
        "stage_summaries": stage_summaries,
        "stage_started_at": _now_iso(),
        "stage_duration_seconds": round(time.perf_counter() - stage_started_at, 6),
        "llm_usage": dict(manifest.llm_usage),
    }


def run_staged(
    smiles: str,
    target_id: str | None,
    constraint: str | None,
    strategy_count: int = 3,
    route_step_limit: int = 25,
    route_proposal_attempts: int = 2,
    route_temperature: float = 0.3,
    edit_iteration_limit: int = 6,
    thinking_level: str | None = "low",
    output_base: str | Path = "runs",
    seed_strategies: list[str] | None = None,
) -> dict[str, Any]:
    project_root = SCRIPTS_SRC.parent
    safe_id = _safe_name(target_id or smiles or "target")
    output_dir = Path(output_base) / safe_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "execution.log"

    _log_event(log_path, {"event": "run_start", "target_smiles": smiles, "target_id": target_id, "output_dir": str(output_dir)})

    api_key = _env_api_key()
    if not api_key:
        error = RuntimeError("Solar/Upstage API key not found. Checked: SOLAR_API_KEY, OPENCODE_SOLAR_KEY, UPSTAGE_API_KEY, OPENCODE_API_KEY.")
        _log_event(log_path, {"event": "run_failed_no_key", "error": repr(error)})
        return _write_report(output_dir, "failed_no_key", None, None, error, [], [])

    _clean_run_dir(output_dir)
    _log_event(log_path, {"event": "run_dir_ready", "output_dir": str(output_dir)})

    try:
        config = _configure_config(
            strategy_count=strategy_count,
            route_step_limit=route_step_limit,
            route_proposal_attempts=route_proposal_attempts,
            route_temperature=route_temperature,
            edit_iteration_limit=edit_iteration_limit,
            thinking_level=thinking_level,
        )
    except Exception as exc:
        _log_event(log_path, {"event": "config_error", "error": repr(exc), "traceback": traceback.format_exc()})
        return _write_report(output_dir, "failed_config", None, None, exc, [], [])

    pipeline, recording_llm = _make_pipeline(config, api_key)
    stage_results: list[dict[str, Any]] = []
    failed_stage: str | None = None
    error: Exception | None = None
    manifest: RunManifest | None = None
    status = "running"

    # 1단계: 전략 생성
    _log_event(log_path, {"event": "stage_strategies_start"})
    strategies_result = _run_stage(
        pipeline,
        output_dir,
        log_path,
        target_smiles=smiles,
        target_id=target_id,
        constraint=constraint,
        stages=("strategies",),
        stage_started_at=time.perf_counter(),
        resume=False,
        seed_strategies=seed_strategies,
    )
    stage_results.append({"name": "strategies", "result": strategies_result})
    if strategies_result["status"] == "failed":
        error = RuntimeError(strategies_result["error_message"])
        failed_stage = "strategies"
        status = "failed"
        _log_event(log_path, {"event": "stage_strategies_failed", "error": strategies_result["error_message"]})
    elif strategies_result["completed_stages"] and "strategies" in strategies_result["completed_stages"]:
        manifest = ArtifactStore(output_dir).load_manifest()
        _log_event(log_path, {"event": "stage_strategies_ok", "run_id": manifest.run_id, "summary": strategies_result["stage_summaries"][0]})
    else:
        failed_stage = "strategies"
        status = "failed"
        error = RuntimeError("strategies stage did not complete as expected")
        _log_event(log_path, {"event": "stage_strategies_unexpected", "result": strategies_result})

    # 전략을 실제로 읽을 수 있는지 확인
    strategies_available = False
    strategies_content = None
    if status != "failed" and manifest is not None:
        try:
            store = ArtifactStore(output_dir)
            strategies_content = store.read_checkpoint(manifest.artifacts["strategies"])
            strategies_available = True
            _log_event(log_path, {"event": "strategies_read_ok", "strategy_count": len(strategies_content.get("strategies", []))})
        except Exception as exc:
            _log_event(log_path, {"event": "strategies_read_failed", "error": repr(exc)})
            failed_stage = "strategies"
            status = "failed"
            error = exc

    # 2단계: 경로 생성 (전략 성공 시에만)
    if status != "failed" and strategies_available:
        _log_event(log_path, {"event": "stage_routes_start"})
        routes_result = _run_stage(
            pipeline,
            output_dir,
            log_path,
            target_smiles=smiles,
            target_id=target_id,
            constraint=constraint,
            stages=("routes",),
            stage_started_at=time.perf_counter(),
            resume=True,
        )
        stage_results.append({"name": "routes", "result": routes_result})
        if routes_result["status"] == "failed":
            error = RuntimeError(routes_result["error_message"])
            failed_stage = "routes"
            status = "failed"
            _log_event(log_path, {"event": "stage_routes_failed", "error": routes_result["error_message"]})
        elif routes_result["completed_stages"] and "routes" in routes_result["completed_stages"]:
            manifest = ArtifactStore(output_dir).load_manifest()
            _log_event(log_path, {"event": "stage_routes_ok", "run_id": manifest.run_id, "summary": routes_result["stage_summaries"][0]})
        else:
            failed_stage = "routes"
            status = "failed"
            error = RuntimeError("routes stage did not complete as expected")
            _log_event(log_path, {"event": "stage_routes_unexpected", "result": routes_result})

    # 실패해도 디스크의 manifest가 권위 있는 상태이므로 항상 다시 읽는다.
    try:
        if (output_dir / "manifest.json").is_file():
            manifest = ArtifactStore(output_dir).load_manifest()
            if status != "failed":
                status = manifest.status or "partial"
    except Exception as exc:
        _log_event(log_path, {"event": "manifest_reload_failed", "error": redact_sensitive_text(str(exc))})

    report = _write_report(output_dir, status, manifest, failed_stage, error, recording_llm.calls, stage_results, seed_strategies)
    _log_event(log_path, {"event": "run_report_written", "status": status, "failed_stage": failed_stage})

    logger.info("status: %s", status)
    logger.info("completed_stages: %s", report["completed_stages"])
    logger.info("failed_stage: %s", failed_stage)
    logger.info("run report: %s", output_dir / "run-report.json")
    logger.info("execution log: %s", log_path)
    return report


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="SynthEx + Solar Pro 4 staged 실행 및 보고서 생성")
    parser.add_argument("--smiles", required=True, help="대상 물질 SMILES")
    parser.add_argument("--target-id", default=None, help="대상 ID")
    parser.add_argument("--constraint", default=None, help="탐구 제약/지시")
    parser.add_argument("--strategy-count", type=int, default=3)
    parser.add_argument("--route-step-limit", type=int, default=25)
    parser.add_argument("--route-proposal-attempts", type=int, default=2)
    parser.add_argument("--route-temperature", type=float, default=0.3)
    parser.add_argument("--edit-iteration-limit", type=int, default=6)
    parser.add_argument("--thinking-level", default="low", help="minimal/low/medium/high")
    parser.add_argument("--output-base", default="runs", help="결과 저장 기본 디렉터리")
    parser.add_argument("--seed-strategy", action="append", default=None, help="이미 생성된 전략 문자열을 그대로 시드로 사용 (반복 가능)")
    args = parser.parse_args()

    report = run_staged(
        smiles=args.smiles,
        target_id=args.target_id,
        constraint=args.constraint,
        strategy_count=args.strategy_count,
        route_step_limit=args.route_step_limit,
        route_proposal_attempts=args.route_proposal_attempts,
        route_temperature=args.route_temperature,
        edit_iteration_limit=args.edit_iteration_limit,
        thinking_level=args.thinking_level,
        output_base=args.output_base,
        seed_strategies=args.seed_strategy,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
