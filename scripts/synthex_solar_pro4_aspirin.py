"""
SynthEx aspirin 탐색 — Solar Pro 4 (Upstage) 연결용 진입 스크립트.

이 스크립트는 SynthEx의 파이프라인·에이전트 계약을 그대로 사용하되,
LLM 공급자만 GeminiClient 대신 SolarClient로 교체한다.

작업 요약
--------
1) 아스피린(SMILES CC(=O)Oc1ccccc1C(=O)O)의 전략(strategy)을 생성
2) 전략별로 ReactionJSON 경로(route)를 구축
3) Critic + Editor 루프로 경로를 검토/수정
4) Analyst로 다차원 분석 수행
5) (환경에 따라 선택) AiZynthFinder로 비재고 리프 완성(stitch)
6) 모든 단계 결과물을 runs/<target_id>/ 아래에 checkpoint(ArtifactStore)로 저장
7) 실행 성공/실패, 호출 모델, 호출 횟수, 토큰 사용량, 에러 추적을
   run-report.json과 execution.log로 남김

환경 변수
----------
이 스크립트는 SynthEx의 기존 llm 계약을 준수하므로, API 키를 코드에
하드코딩하지 않는다. 다음 중 하나로 키를 받는다.

- SOLAR_API_KEY: OpenAI 호환 엔드포인트용 키 (권장)
- OPENCODE_SOLAR_KEY: OpenCode 설정에 이미 존재한다고 가정하는 키를
  넘겨받을 때 사용하는 대체 이름 (문서상 편의용)
- 혹은 실행 시점에 SolarClient(api_key=...)를 명시적으로 전달

SynthEx의 기본 ManuscriptConfig는 provider=gemini로 고정되어 있으므로,
이 스크립트는 ConfigModel을 직접 구성해 provider/model 필드를 Solar 호출에
맞게 덮어쓴다(manifest의 provider/model 기록은 Solar Pro 4로 남긴다).

엔드포인트
----------
OpenCode 설정의 Upstage 제공자 baseURL을 사용한다:
  https://api.upstage.ai/v1

SolarClient는 이 base_url과 api_key를 사용해 OpenAI 호환 Chat Completions
엔드포인트로 호출한다. models.generate_content-style responses API는 이
어댑터의 기본 경로가 아니므로 사용하지 않는다.

이 스크립트는 아래 어댑터 모듈을 재사용한다:
  from synthex_adapters.solar import SolarClient

신뢰 경계
----------
- 이 스크립트는 SynthEx 소스 파일(agents, pipeline, llm 계약 등)을 수정하지
  않는다. SolarClient는 새로운 adapter 모듈로 분리한다.
- AiZynthFinder 완성(stitched_routes)은 선택적 단계로, 관련 설정·자산이
  없으면 건너뛴다.

출력 디렉터리 예시
-------------------
runs/aspirin-solar-pro4/
  manifest.json
  strategies.json
  routes.json
  routes.tree.json
  reviewed_routes.json
  reviewed_routes.tree.json
  analyses.json
  stitched_routes.json (완성 단계가 활성화된 경우)
  run-report.json
  execution.log
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from synthex.agents import (
    Analyst,
    Critic,
    Editor,
    RouteBuilder,
    StrategyGenerator,
)
from synthex.agents.base import AgentContext
from synthex.agents.models import (
    AnalysisResult,
    AnalystRequest,
    CriticRequest,
    Critique,
    EditorRequest,
    EditorResult,
    RouteBuildRequest,
    RouteBuildResult,
    StrategyRequest,
    StrategyResult,
)
from synthex.artifacts import (
    ArtifactStore,
    Failure,
    RunManifest,
    code_revision,
    dependency_versions,
    digest_file,
    digest_json,
    now_utc,
    redact_sensitive_text,
)
from synthex.llm import LLMRequest, LLMResponse, LLMUsage, Message, ToolCall, ToolSpec
from synthex.pipeline import PipelineComponents, _UsageRecorder
from synthex.routejson import Route
from synthex.routejson.strategy_tree import op_plan_to_strategy_tree

# 외부 어댑터 모듈을 현재 스크립트와 같은 레벨에서 import 한다.
try:
    from synthex_adapters.solar import SolarClient
except ImportError as exc:
    # 스크립트가 단독으로 실행될 때 어댑터가 같은 디렉터리 하위에 없으면
    # 이 경로를 추가해 찾는다.
    adapter_dir = Path(__file__).resolve().parent / "synthex_adapters"
    if adapter_dir.is_dir():
        sys.path.insert(0, str(adapter_dir.parent))
    from synthex_adapters.solar import SolarClient

# ---------------------------------------------------------------------------
# 설정값 — 스크립트 상단에서 한 번에 조정할 수 있도록 노출
# ---------------------------------------------------------------------------

TARGET_SMILES = "CC(=O)Oc1ccccc1C(=O)O"
TARGET_ID = "aspirin-demo"
OUTPUT_DIR = Path("runs/aspirin-solar-pro4")

# Solar Pro 4 호출 파라미터
SOLAR_MODEL = "solar-pro4"
SOLAR_BASE_URL = "https://api.upstage.ai/v1"
SOLAR_TIMEOUT_SECONDS = 600
SOLAR_MAX_ATTEMPTS = 3

# SynthEx 에이전트 파라미터 (필요하면 조정)
STRATEGY_COUNT = 3
ROUTE_STEP_LIMIT = 25
ROUTE_PROPOSAL_ATTEMPTS = 3
EDIT_ITERATION_LIMIT = 6
EDITOR_TURN_LIMIT = 60

# 온도/씽킹 레벨은 SynthEx의 기본값을 따르되, 필요하면 여기서 덮어쓴다.
STRATEGY_TEMPERATURE = 0.1
ROUTE_TEMPERATURE = 0.3
CRITIC_TEMPERATURE = 0.1
EDITOR_TEMPERATURE = 0.3
ANALYST_TEMPERATURE = 0.3

STRATEGY_THINKING_LEVEL = None
ROUTE_THINKING_LEVEL = "medium"
CRITIC_THINKING_LEVEL = "medium"
EDITOR_THINKING_LEVEL = "medium"
ANALYST_THINKING_LEVEL = None

# 완성(stitching) 단계 활성화 여부. 관련 설정/자산이 없으면 False로 둔다.
ENABLE_COMPLETION = False

# ---------------------------------------------------------------------------
# 로그 구성
# ---------------------------------------------------------------------------

logger = logging.getLogger("synthex-solar-aspirin")

def _configure_logging(log_path: Path) -> None:
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Manifest / 구성 헬퍼
# ---------------------------------------------------------------------------

def _default_config() -> dict[str, Any]:
    return {
        "format_version": "synthex-config-1.0",
        "profile": "manuscript",
        "llm": {
            "provider": "solar",
            "model": SOLAR_MODEL,
            "use_vertex_ai": False,
            "timeout_seconds": SOLAR_TIMEOUT_SECONDS,
            "max_attempts": SOLAR_MAX_ATTEMPTS,
            "thinking_level": None,
        },
        "agents": {
            "strategy_count": STRATEGY_COUNT,
            "route_step_limit": ROUTE_STEP_LIMIT,
            "route_proposal_attempts": ROUTE_PROPOSAL_ATTEMPTS,
            "edit_iteration_limit": EDIT_ITERATION_LIMIT,
            "editor_turn_limit": EDITOR_TURN_LIMIT,
            "strategy_temperature": STRATEGY_TEMPERATURE,
            "route_temperature": ROUTE_TEMPERATURE,
            "critic_temperature": CRITIC_TEMPERATURE,
            "editor_temperature": EDITOR_TEMPERATURE,
            "analyst_temperature": ANALYST_TEMPERATURE,
            "strategy_thinking_level": STRATEGY_THINKING_LEVEL,
            "route_thinking_level": ROUTE_THINKING_LEVEL,
            "critic_thinking_level": CRITIC_THINKING_LEVEL,
            "editor_thinking_level": EDITOR_THINKING_LEVEL,
            "analyst_thinking_level": ANALYST_THINKING_LEVEL,
        },
        "completion": {
            "enabled": ENABLE_COMPLETION,
            "aizynthfinder_config": None,
            "stocks": None,
            "expansion_policies": None,
            "filter_policies": None,
            "budget": {
                "max_transforms": 6,
                "iteration_limit": 500,
                "time_limit": 1200,
                "return_first": True,
            },
        },
        "external_asset_manifest": None,
    }


def _build_components(llm: SolarClient) -> PipelineComponents:
    return PipelineComponents(
        llm=llm,
        strategy_generator=StrategyGenerator(
            temperature=STRATEGY_TEMPERATURE,
            thinking_level=STRATEGY_THINKING_LEVEL,
        ),
        route_builder=RouteBuilder(
            temperature=ROUTE_TEMPERATURE,
            thinking_level=ROUTE_THINKING_LEVEL,
        ),
        critic=Critic(
            temperature=CRITIC_TEMPERATURE,
            thinking_level=CRITIC_THINKING_LEVEL,
        ),
        editor=Editor(
            temperature=EDITOR_TEMPERATURE,
            thinking_level=EDITOR_THINKING_LEVEL,
        ),
        analyst=Analyst(
            temperature=ANALYST_TEMPERATURE,
            thinking_level=ANALYST_THINKING_LEVEL,
        ),
        leaf_completer=None,
    )


def _manifest_from_components(
    target: str,
    target_id: str | None,
    constraint: str | None,
) -> RunManifest:
    config_dump = _default_config()
    input_value = {"target": target, "target_id": target_id, "constraint": constraint}
    lock = Path(__file__).parents[2] / "uv.lock"
    packaged_lock = Path(__file__).with_name("uv.lock")
    lock_path = lock if lock.is_file() else packaged_lock

    prompt_versions = {
        "strategy_generator": "strategy-generator-v1",
        "route_builder": "route-builder-reactionjson-v2",
        "critic": "critic-v2-atom-accounting",
        "editor": "editor-v1",
        "analyst": "analyst-v1",
    }
    prompt_root = Path(__file__).resolve().parent.parent / "src" / "synthex" / "prompts"
    prompt_sha256 = {}
    for name, filename in {
        "strategy_generator": "strategy_generator_v1.py",
        "route_builder": "route_builder_v1.py",
        "critic": "critic_v1.py",
        "editor": "editor_v1.py",
        "analyst": "analyst_v1.py",
    }.items():
        prompt_file = prompt_root / filename
        if prompt_file.is_file():
            prompt_sha256[name] = digest_file(prompt_file)
        else:
            prompt_sha256[name] = None

    created = now_utc()
    manifest = RunManifest(
        run_id=digest_json({"input": input_value, "config": config_dump})[:16],
        target=target,
        target_id=target_id,
        constraint=constraint,
        input_sha256=digest_json(input_value),
        config=config_dump,
        config_sha256=digest_json(config_dump),
        code_revision=code_revision(),
        dependency_versions=dependency_versions(),
        dependency_lock_sha256=digest_file(lock_path) if lock_path.is_file() else None,
        provider="solar",
        model=SOLAR_MODEL,
        prompt_versions=prompt_versions,
        prompt_sha256=prompt_sha256,
        external_assets={},
        created_at=created,
        updated_at=created,
    )
    return manifest


def _checkpoint(
    store: ArtifactStore,
    manifest: RunManifest,
    stage: str,
    payload: Any,
) -> None:
    """ArtifactStore 기반 checkpoint 저장 + 매니페스트 갱신.

    SynthEx의 SynthexPipeline._checkpoint과 동일한 구조를 따르되,
    이 스크립트에서는 stage_config를 고정 구성으로 재기록해 digest chaining을
    유지한다.
    """
    stage_config = _default_config()
    manifest.artifacts[stage] = store.write_checkpoint(stage, payload, None)
    manifest.stage_configs[stage] = stage_config
    manifest.stage_config_sha256[stage] = digest_json(stage_config)
    if stage == "stitched_routes":
        manifest.external_assets = {}
    manifest.completed_stages.append(stage)
    manifest.current_stage = stage
    manifest.status = "running"
    manifest.failure = None
    store.save_manifest(manifest)


def _record_runtime(
    store: ArtifactStore,
    manifest: RunManifest,
    recorder: _UsageRecorder,
    stage: str,
    started: float,
) -> None:
    manifest.llm_usage = recorder.snapshot()
    manifest.stage_durations_seconds[stage] = round(time.perf_counter() - started, 6)
    store.save_manifest(manifest)


def _write_trees(store: ArtifactStore, stage: str, payload: list[dict[str, Any]]) -> None:
    from synthex.routejson import Route as _Route
    from synthex.routejson.executor import op_plan_to_tree

    trees = []
    for item in payload:
        if "route" not in item:
            trees.append(item)
            continue
        tree = op_plan_to_tree(_Route.model_validate(item["route"]).to_op_plan())
        if item.get("strategy"):
            tree = {"strategy_prompt": item["strategy"]} | tree
        trees.append(tree)
    (store.root / f"{stage}.tree.json").write_text(
        json.dumps(trees, indent=2, ensure_ascii=False) + "\n"
    )


# ---------------------------------------------------------------------------
# 단일 타겟 파이프라인 실행 (에이전트 직접 호출)
# ---------------------------------------------------------------------------

async def run_aspirin() -> dict[str, Any]:
    """아스피린 탐색을 실행하고 결과 메타를 반환한다."""
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "execution.log"
    _configure_logging(log_path)

    api_key: str | None = None
    if "SOLAR_API_KEY" in os.environ:
        api_key = os.environ["SOLAR_API_KEY"]
    elif "OPENCODE_SOLAR_KEY" in os.environ:
        api_key = os.environ["OPENCODE_SOLAR_KEY"]

    llm = SolarClient(
        api_key=api_key,
        model=SOLAR_MODEL,
        base_url=SOLAR_BASE_URL,
        timeout_seconds=SOLAR_TIMEOUT_SECONDS,
        max_attempts=SOLAR_MAX_ATTEMPTS,
    )
    components = _build_components(llm)
    recorder = _UsageRecorder(llm, {})
    context = AgentContext(llm=recorder, run_id="aspirin-solar-pro4")

    store = ArtifactStore(output)
    manifest = _manifest_from_components(TARGET_SMILES, TARGET_ID, None)
    if store.manifest_path.exists():
        raise RuntimeError(f"출력 디렉터리가 이미 존재합니다: {store.root}")
    store.save_manifest(manifest)

    result: dict[str, Any] = {
        "target": TARGET_SMILES,
        "target_id": TARGET_ID,
        "status": "failed",
        "provider": "solar",
        "model": SOLAR_MODEL,
        "endpoint": SOLAR_BASE_URL,
        "stages": [],
        "llm_calls": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "thought_tokens": 0,
        "total_tokens": 0,
        "artifacts": [],
        "failure": None,
    }

    stage_started: float | None = None
    active_stage = "created"

    def _fail_stage(stage: str, exc: Exception) -> None:
        nonlocal active_stage
        if stage_started is not None and stage not in manifest.stage_durations_seconds:
            manifest.stage_durations_seconds[stage] = round(time.perf_counter() - stage_started, 6)
        manifest.status = "failed"
        manifest.current_stage = stage
        manifest.failure = Failure(
            stage=stage,
            category=type(exc).__name__,
            message=redact_sensitive_text(str(exc)),
        )
        manifest.llm_usage = recorder.snapshot()
        store.save_manifest(manifest)
        result["status"] = "failed"
        result["failure"] = {
            "category": type(exc).__name__,
            "message": redact_sensitive_text(str(exc)),
        }
        logger.exception("파이프라인 실행 실패")

    try:
        # ---- strategies ----
        active_stage = "strategies"
        stage_started = time.perf_counter()
        strategy_result = await components.strategy_generator.run(
            StrategyRequest(target=TARGET_SMILES, count=STRATEGY_COUNT, constraint=None),
            context,
        )
        strategies_payload = strategy_result.model_dump(mode="json") | {"origin": "generated"}
        _checkpoint(store, manifest, "strategies", strategies_payload)
        _record_runtime(store, manifest, recorder, "strategies", stage_started)
        result["stages"].append("strategies")

        (store.root / "strategies.tree.json").write_text(
            json.dumps([{"strategy": s} for s in strategy_result.strategies], indent=2) + "\n"
        )

        # ---- routes ----
        active_stage = "routes"
        stage_started = time.perf_counter()
        strategies = store.read_checkpoint(manifest.artifacts["strategies"])["strategies"]
        routes: list[dict[str, Any]] = []
        for strategy in strategies:
            try:
                route_result = await components.route_builder.run(
                    RouteBuildRequest(
                        target=TARGET_SMILES,
                        strategy=strategy,
                        max_steps=ROUTE_STEP_LIMIT,
                        proposal_attempts=ROUTE_PROPOSAL_ATTEMPTS,
                    ),
                    context,
                )
                routes.append(
                    {
                        "status": "ok",
                        "strategy": strategy,
                        "termination": route_result.termination,
                        "route": route_result.route.model_dump(mode="json"),
                    }
                )
            except Exception as exc:
                routes.append(
                    {
                        "status": "failed",
                        "strategy": strategy,
                        "failure": {
                            "category": type(exc).__name__,
                            "message": redact_sensitive_text(str(exc)),
                        },
                    }
                )
        if not any(item.get("status") == "ok" for item in routes):
            raise ValueError("모든 경로 구축이 실패했습니다")
        _checkpoint(store, manifest, "routes", routes)
        _write_trees(store, "routes", routes)
        _record_runtime(store, manifest, recorder, "routes", stage_started)
        result["stages"].append("routes")

        # ---- reviewed_routes (Critic + Editor 루프) ----
        active_stage = "reviewed_routes"
        stage_started = time.perf_counter()
        routes_payload = store.read_checkpoint(manifest.artifacts["routes"])
        reviewed: list[dict[str, Any]] = []
        for item in routes_payload:
            if item.get("status") != "ok" or "route" not in item:
                reviewed.append(item)
                continue
            try:
                route = Route.model_validate(item["route"])
                iterations: list[dict[str, Any]] = []
                snapshots = {0: route}
                critique: Critique | None = None
                if EDIT_ITERATION_LIMIT > 0:
                    critique = await components.critic.run(
                        CriticRequest(route=route, constraint=None), context
                    )
                    for iteration in range(1, EDIT_ITERATION_LIMIT + 1):
                        if critique.is_resolved:
                            break
                        edited = await components.editor.run(
                            EditorRequest(
                                route=route,
                                critique=critique,
                                strategy=item["strategy"],
                                iteration=iteration,
                                max_iterations=EDIT_ITERATION_LIMIT,
                                max_turns=EDITOR_TURN_LIMIT,
                                history=iterations,
                                snapshots=snapshots,
                            ),
                            context,
                        )
                        iterations.append(
                            {
                                "iteration": iteration,
                                "critique": critique.model_dump(mode="json"),
                                "edit_log": edited.edit_log,
                                "tool_history": [
                                    record.model_dump(mode="json") for record in edited.tool_history
                                ],
                            }
                        )
                        route = edited.route
                        snapshots[iteration] = route
                        critique = await components.critic.run(
                            CriticRequest(route=route, constraint=None), context
                        )
                reviewed.append(
                    {
                        "status": "ok",
                        "strategy": item["strategy"],
                        "route": route.model_dump(mode="json"),
                        "final_critique": critique.model_dump(mode="json") if critique is not None else None,
                        "iterations": iterations,
                    }
                )
            except Exception as exc:
                reviewed.append(
                    {
                        "status": "review_failed",
                        "strategy": item.get("strategy", ""),
                        "route": item.get("route", {}),
                        "final_critique": None,
                        "iterations": [],
                        "failure": {
                            "category": type(exc).__name__,
                            "message": redact_sensitive_text(str(exc)),
                        },
                    }
                )
        if not any(item.get("status") == "ok" for item in reviewed):
            raise ValueError("모든 경로 검토가 실패했습니다")
        _checkpoint(store, manifest, "reviewed_routes", reviewed)
        _write_trees(store, "reviewed_routes", reviewed)
        _record_runtime(store, manifest, recorder, "reviewed_routes", stage_started)
        result["stages"].append("reviewed_routes")

        # ---- analyses ----
        active_stage = "analyses"
        stage_started = time.perf_counter()
        reviewed_payload = store.read_checkpoint(manifest.artifacts["reviewed_routes"])
        analyses: list[dict[str, Any]] = []
        for item in reviewed_payload:
            if "route" not in item:
                analyses.append(item)
                continue
            try:
                analysis = await components.analyst.run(
                    AnalystRequest(
                        route=Route.model_validate(item["route"]),
                        constraint=None,
                    ),
                    context,
                )
                analyses.append(analysis.model_dump(mode="json") | {"status": "ok"})
            except Exception as exc:
                analyses.append(
                    {
                        "status": "failed",
                        "strategy": item.get("strategy", ""),
                        "failure": {
                            "category": type(exc).__name__,
                            "message": redact_sensitive_text(str(exc)),
                        },
                    }
                )
        if not any(item.get("status") == "ok" for item in analyses):
            raise ValueError("모든 분석이 실패했습니다")
        _checkpoint(store, manifest, "analyses", analyses)
        _record_runtime(store, manifest, recorder, "analyses", stage_started)
        result["stages"].append("analyses")

        # ---- stitched_routes (선택적) ----
        if ENABLE_COMPLETION:
            active_stage = "stitched_routes"
            stage_started = time.perf_counter()
            stitched: list[dict[str, Any]] = []
            for item in reviewed_payload:
                if "route" not in item:
                    stitched.append(item)
                    continue
                try:
                    # 이 데모에서는 AiZynthFinder 완성 설정이 없으므로,
                    # 완성 단계가 켜져도 즉시 실패 처리되도록 한다.
                    raise RuntimeError(
                        "ENABLE_COMPLETION=True이지만 AiZynthFinder 설정/자산이 없습니다. "
                        "configs/synthex.manuscript.yml의 completion.aizynthfinder_config를 "
                        "설정하거나 ENABLE_COMPLETION=False로 두세요."
                    )
                except Exception as exc:
                    stitched.append(
                        {
                            "status": "failed",
                            "strategy": item.get("strategy", ""),
                            "failure": {
                                "category": type(exc).__name__,
                                "message": redact_sensitive_text(str(exc)),
                            },
                        }
                    )
            _checkpoint(store, manifest, "stitched_routes", stitched)
            _record_runtime(store, manifest, recorder, "stitched_routes", stage_started)
            result["stages"].append("stitched_routes")

        manifest.status = "complete"
        manifest.current_stage = "complete"
        manifest.llm_usage = recorder.snapshot()
        store.save_manifest(manifest)
        result["status"] = "complete"

    except Exception as exc:
        _fail_stage(active_stage, exc)
        raise
    finally:
        result["llm_calls"] = recorder.snapshot().get("calls", 0)
        usage = recorder.snapshot()
        result["input_tokens"] = usage.get("input_tokens", 0)
        result["cached_input_tokens"] = usage.get("cached_input_tokens", 0)
        result["output_tokens"] = usage.get("output_tokens", 0)
        result["thought_tokens"] = usage.get("thought_tokens", 0)
        result["total_tokens"] = usage.get("total_tokens", 0)
        result["artifacts"] = list(manifest.artifacts.keys())

        report_path = output / "run-report.json"
        report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        logger.info("run-report.json 저장 완료: %s", report_path)
        manifest.llm_usage = recorder.snapshot()
        store.save_manifest(manifest)

    return result


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        report = asyncio.run(run_aspirin())
    except Exception as exc:
        print(f"synthax-solar-aspirin 실패: {redact_sensitive_text(str(exc))}", file=sys.stderr)
        return 2
    print(f"상태: {report['status']}")
    print(f"대상: {report['target_id']} ({report['target']})")
    print(f"공급자/모델: {report['provider']} / {report['model']}")
    print(f"엔드포인트: {report['endpoint']}")
    print(f"LLM 호출: {report['llm_calls']}회")
    print(f"토큰(input/output/total): {report['input_tokens']} / {report['output_tokens']} / {report['total_tokens']}")
    print(f"완료 단계: {' -> '.join(report['stages'])}")
    print(f"아티팩트: {', '.join(report['artifacts'])}")
    print(f"출력 디렉터리: {Path(OUTPUT_DIR).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
