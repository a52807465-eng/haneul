#!/usr/bin/env python3
"""
SynthEx를 Solar Pro 4(Upstage)로 실행해 합성 경로를 탐색하고,
성공·실패 결과와 실행 기록을 보고서로 남기는 실행 스크립트.

이 스크립트는 SynthEx 소스를 수정하지 않고, 별도 어댑터(scripts/synthex_adapters/solar.py)를
사용해 SolarClient를 SynthEx 파이프라인에 주입한다.

실행 예시:
    python scripts/synthex_solar_pro4_run.py \
        --smiles "CC(=O)Oc1ccccc1C(=O)O" \
        --target-id aspirin-demo \
        --constraint "짧고 상업적으로 구입 가능한 출발 물질을 사용하는 경로를 탐색하라."
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
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
    from synthex.artifacts import ArtifactStore, RunManifest, code_revision, dependency_versions
    from synthex.config import ManuscriptConfig
    from synthex.llm import LLMClient, LLMRequest, LLMResponse, LLMUsage, Message, ThinkingLevel, ToolCall, ToolSpec
    from synthex.pipeline import PipelineComponents, SynthexPipeline
    from synthex.routejson import Route
    from synthex.routejson.render import render_plan
except Exception as exc:
    print("synthex import failed:", exc)
    sys.exit(2)

from synthex_adapters.solar import SolarClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("synthex_solar_pro4_run")


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _safe_target_id(value: str) -> str:
    import re
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not safe:
        safe = "target"
    return safe


def _env_api_key() -> str | None:
    for name in ("SOLAR_API_KEY", "OPENCODE_SOLAR_KEY", "UPSTAGE_API_KEY", "OPENCODE_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _write_run_report(
    output_dir: Path,
    status: str,
    manifest: RunManifest | None,
    error: Exception | None,
    llm_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    report = {
        "run_type": "synthex-solar-pro4",
        "target": "unknown",
        "target_smiles": None,
        "target_id": None,
        "model": "solar-pro4",
        "provider": "upstage",
        "base_url": "https://api.upstage.ai/v1/chat/completions",
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "manifest.json") if manifest is not None else None,
        "completed_stages": list(manifest.completed_stages) if manifest is not None else [],
        "status": status,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "llm_calls": llm_calls,
    }
    if error is not None:
        report["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        }
    if manifest is not None:
        report["target"] = manifest.target or "unknown"
        report["target_smiles"] = manifest.target or None
        report["target_id"] = manifest.target_id or None
        report["llm_usage"] = manifest.llm_usage
        report["stage_durations_seconds"] = manifest.stage_durations_seconds
    (output_dir / "run-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def _log_event(log_path: Path, event: dict[str, Any]) -> None:
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)


def _configure_manifest_config(
    strategy_count: int = 3,
    route_step_limit: int = 25,
    edit_iteration_limit: int = 6,
    thinking_level: ThinkingLevel | None = "low",
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
            "thinking_level": thinking_level,
        },
        agents={
            "strategy_count": strategy_count,
            "route_step_limit": route_step_limit,
            "route_proposal_attempts": 2,
            "edit_iteration_limit": edit_iteration_limit,
            "editor_turn_limit": 80,
            "strategy_temperature": 0.1,
            "route_temperature": 0.3,
            "critic_temperature": 0.1,
            "editor_temperature": 0.3,
            "analyst_temperature": 0.3,
            "strategy_thinking_level": None,
            "route_thinking_level": thinking_level,
            "critic_thinking_level": thinking_level,
            "editor_thinking_level": thinking_level,
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


class _CallRecorder:
    """LLM 호출 이력을 가볍게 기록한다."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError("CallRecorder는 래퍼로만 사용한다.")


def _wrap_with_recorder(client: SolarClient, recorder: _CallRecorder) -> LLMClient:
    class _RecordClient:
        def __init__(self, client: SolarClient, recorder: _CallRecorder) -> None:
            self._client = client
            self._recorder = recorder

        async def generate(self, request: LLMRequest) -> LLMResponse:
            start = _now_iso()
            try:
                response = await self._client.generate(request)
            except Exception as exc:
                self._recorder.calls.append({
                    "status": "error",
                    "started_at": start,
                    "error": repr(exc),
                    "model": getattr(self._client, "model", None),
                })
                raise
            ended_at = _now_iso()
            self._recorder.calls.append({
                "status": "ok",
                "started_at": start,
                "ended_at": ended_at,
                "model": getattr(self._client, "model", None),
                "usage": None if response.usage is None else response.usage.model_dump(),
                "tool_call_count": len(response.tool_calls),
                "text_len": len(response.text),
            })
            return response

    return _RecordClient(client, recorder)


def _build_pipeline(
    config: ManuscriptConfig,
    api_key: str,
) -> SynthexPipeline:
    llm = SolarClient(
        model=config.llm.model,
        api_key=api_key,
        base_url="https://api.upstage.ai/v1",
        timeout_seconds=config.llm.timeout_seconds,
        max_attempts=config.llm.max_attempts,
    )
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
    return SynthexPipeline(config, components)


async def _run_pipeline(
    pipeline: SynthexPipeline,
    target_smiles: str,
    output_dir: Path,
    target_id: str | None,
    constraint: str | None,
) -> RunManifest:
    manifest = await pipeline.run(
        target=target_smiles,
        output=str(output_dir),
        target_id=target_id,
        constraint=constraint,
    )
    return manifest


def run_once(
    smiles: str,
    target_id: str | None,
    constraint: str | None,
    strategy_count: int = 3,
    route_step_limit: int = 25,
    edit_iteration_limit: int = 6,
    thinking_level: str | None = "low",
    output_base: str | Path = "runs",
) -> dict[str, Any]:
    project_root = SCRIPTS_SRC.parent
    output_dir = Path(output_base) / _safe_target_id(target_id or smiles or "target")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "execution.log"

    _log_event(log_path, {"event": "run_start", "target_smiles": smiles, "target_id": target_id})

    api_key = _env_api_key()
    if not api_key:
        error = RuntimeError("Solar/Upstage API key not found in environment. Checked: SOLAR_API_KEY, OPENCODE_SOLAR_KEY, UPSTAGE_API_KEY, OPENCODE_API_KEY.")
        _log_event(log_path, {"event": "run_failed_no_key", "error": repr(error)})
        _write_run_report(output_dir, "failed_no_key", None, error, [])
        logger.error(error)
        return _write_run_report(output_dir, "failed_no_key", None, error, [])

    try:
        config = _configure_manifest_config(
            strategy_count=strategy_count,
            route_step_limit=route_step_limit,
            edit_iteration_limit=edit_iteration_limit,
            thinking_level=thinking_level if thinking_level is not None else "low",
        )
    except Exception as exc:
        error = exc
        _log_event(log_path, {"event": "run_config_error", "error": repr(exc), "traceback": traceback.format_exc()})
        _write_run_report(output_dir, "failed_config", None, exc, [])
        logger.error("config error: %s", exc)
        return _write_run_report(output_dir, "failed_config", None, exc, [])

    recorder = _CallRecorder()
    llm = _wrap_with_recorder(SolarClient(
        model=config.llm.model,
        api_key=api_key,
        base_url="https://api.upstage.ai/v1",
        timeout_seconds=config.llm.timeout_seconds,
        max_attempts=config.llm.max_attempts,
    ), recorder)

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
    pipeline = SynthexPipeline(config, components)

    manifest: RunManifest | None = None
    error: Exception | None = None
    try:
        manifest = asyncio.run(
            _run_pipeline(
                pipeline,
                target_smiles=smiles,
                output_dir=output_dir,
                target_id=target_id,
                constraint=constraint,
            )
        )
        _log_event(log_path, {"event": "run_complete", "status": manifest.status, "run_id": manifest.run_id})
    except Exception as exc:
        error = exc
        _log_event(log_path, {"event": "run_failed", "error": repr(exc), "traceback": traceback.format_exc()})
        logger.error("pipeline failed: %s", exc)

    report = _write_run_report(output_dir, (manifest.status or "failed") if manifest is not None else "failed", manifest, error, recorder.calls)
    _log_event(log_path, {"event": "run_report_written", "status": report["status"]})

    if manifest is not None:
        logger.info("manifest: %s", output_dir / "manifest.json")
        logger.info("completed stages: %s", manifest.completed_stages)
    logger.info("run report: %s", output_dir / "run-report.json")
    logger.info("execution log: %s", log_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="SynthEx + Solar Pro 4 실행 및 보고서 생성")
    parser.add_argument("--smiles", required=True, help="대상 물질 SMILES")
    parser.add_argument("--target-id", default=None, help="대상 ID (출력 디렉터리 이름에 사용)")
    parser.add_argument("--constraint", default=None, help="탐구 제약/지시")
    parser.add_argument("--strategy-count", type=int, default=3)
    parser.add_argument("--route-step-limit", type=int, default=25)
    parser.add_argument("--edit-iteration-limit", type=int, default=6)
    parser.add_argument("--thinking-level", default="low", help="minimal/low/medium/high")
    parser.add_argument("--output-base", default="runs", help="결과 저장 기본 디렉터리")
    args = parser.parse_args()

    report = run_once(
        smiles=args.smiles,
        target_id=args.target_id,
        constraint=args.constraint,
        strategy_count=args.strategy_count,
        route_step_limit=args.route_step_limit,
        edit_iteration_limit=args.edit_iteration_limit,
        thinking_level=args.thinking_level,
        output_base=args.output_base,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
