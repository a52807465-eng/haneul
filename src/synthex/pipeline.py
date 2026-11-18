"""Resumable application pipeline for the five manuscript agents."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from synthex.completion import LeafCompleter, complete_route
from synthex.config import ExternalAssetManifest, ManuscriptConfig
from synthex.llm import LLMClient, LLMRequest, LLMResponse
from synthex.routejson import Route
from synthex.routejson.strategy_tree import op_plan_to_strategy_tree

STAGES = ("strategies", "routes", "reviewed_routes", "analyses", "stitched_routes")
STAGE_DEPENDENCIES = {
    "strategies": (),
    "routes": ("strategies",),
    "reviewed_routes": ("routes",),
    "analyses": ("reviewed_routes",),
    "stitched_routes": ("reviewed_routes",),
}


def _failure(strategy: str, exc: Exception) -> dict[str, Any]:
    """Record one item's failure so the rest of the stage can still complete."""

    return {
        "status": "failed",
        "strategy": strategy,
        "failure": {
            "category": type(exc).__name__,
            "message": redact_sensitive_text(str(exc)),
        },
    }


def _succeeded(item: dict[str, Any]) -> bool:
    """Items written before per-item isolation carry no status and are successes."""

    return item.get("status", "ok") == "ok"


def _require_one_success(items: list[dict[str, Any]], stage: str) -> None:
    if items and not any(_succeeded(item) for item in items):
        first = items[0].get("failure", {})
        raise ValueError(
            f"every item in stage {stage!r} failed; first failure was "
            f"{first.get('category', 'unknown')}: {first.get('message', '')}"
        )


class _UsageRecorder:
    def __init__(self, client: LLMClient, initial: dict[str, int]) -> None:
        self.client = client
        self.values = {
            "calls": int(initial.get("calls", 0)),
            "input_tokens": int(initial.get("input_tokens", 0)),
            "cached_input_tokens": int(initial.get("cached_input_tokens", 0)),
            "output_tokens": int(initial.get("output_tokens", 0)),
            "thought_tokens": int(initial.get("thought_tokens", 0)),
            "total_tokens": int(initial.get("total_tokens", 0)),
        }

    async def generate(self, request: LLMRequest) -> LLMResponse:
        response = await self.client.generate(request)
        self.values["calls"] += 1
        if response.usage is not None:
            self.values["input_tokens"] += response.usage.input_tokens or 0
            self.values["cached_input_tokens"] += response.usage.cached_input_tokens or 0
            self.values["output_tokens"] += response.usage.output_tokens or 0
            self.values["thought_tokens"] += response.usage.thought_tokens or 0
            self.values["total_tokens"] += response.usage.total_tokens or 0
        return response

    def snapshot(self) -> dict[str, int]:
        return dict(self.values)


@dataclass(frozen=True)
class PipelineComponents:
    llm: LLMClient
    strategy_generator: StrategyGenerator
    route_builder: RouteBuilder
    critic: Critic
    editor: Editor
    analyst: Analyst
    leaf_completer: LeafCompleter | None = None


class SynthexPipeline:
    def __init__(self, config: ManuscriptConfig, components: PipelineComponents) -> None:
        self.config = config
        self.components = components

    def _new_manifest(
        self,
        target: str,
        target_id: str | None,
        constraint: str | None,
    ) -> RunManifest:
        config_dump = self.config.model_dump(mode="json")
        input_value = {"target": target, "target_id": target_id, "constraint": constraint}
        source_lock = Path(__file__).parents[2] / "uv.lock"
        packaged_lock = Path(__file__).with_name("uv.lock")
        lock = source_lock if source_lock.is_file() else packaged_lock
        prompt_versions = {
            agent.name: agent.prompt_version
            for agent in (
                self.components.strategy_generator,
                self.components.route_builder,
                self.components.critic,
                self.components.editor,
                self.components.analyst,
            )
        }
        prompt_files = {
            "strategy_generator": "strategy_generator_v1.py",
            "route_builder": "route_builder_v1.py",
            "critic": "critic_v1.py",
            "editor": "editor_v1.py",
            "analyst": "analyst_v1.py",
        }
        prompt_root = Path(__file__).parent / "prompts"
        prompt_sha256 = {
            name: digest_file(prompt_root / filename) for name, filename in prompt_files.items()
        }
        external_assets = self._external_asset_digests()
        created = now_utc()
        return RunManifest(
            run_id=digest_json({"input": input_value, "config": config_dump})[:16],
            target=target,
            target_id=target_id,
            constraint=constraint,
            input_sha256=digest_json(input_value),
            config=config_dump,
            config_sha256=digest_json(config_dump),
            code_revision=code_revision(),
            dependency_versions=dependency_versions(),
            dependency_lock_sha256=digest_file(lock) if lock.is_file() else None,
            provider=self.config.llm.provider,
            model=self.config.llm.model,
            prompt_versions=prompt_versions,
            prompt_sha256=prompt_sha256,
            external_assets=external_assets,
            created_at=created,
            updated_at=created,
        )

    def _external_asset_digests(self) -> dict[str, str]:
        external_assets: dict[str, str] = {}
        for key, path in (
            ("aizynthfinder_config", self.config.completion.aizynthfinder_config),
            ("asset_manifest", self.config.external_asset_manifest),
        ):
            if path is not None and path.is_file():
                external_assets[key] = digest_file(path)
        if (
            self.config.external_asset_manifest is not None
            and self.config.external_asset_manifest.is_file()
        ):
            asset_manifest = ExternalAssetManifest.model_validate_json(
                self.config.external_asset_manifest.read_text()
            )
            for asset in asset_manifest.assets:
                external_assets[f"asset:{asset.identifier}"] = asset.sha256
        return external_assets

    @staticmethod
    def _parent_digest(manifest: RunManifest) -> str | None:
        if not manifest.completed_stages:
            return None
        return manifest.artifacts[manifest.completed_stages[-1]].sha256

    def _checkpoint(
        self,
        store: ArtifactStore,
        manifest: RunManifest,
        stage: str,
        payload: Any,
    ) -> None:
        reference = store.write_checkpoint(stage, payload, self._parent_digest(manifest))
        stage_config = self.config.model_dump(mode="json")
        manifest.artifacts[stage] = reference
        manifest.stage_configs[stage] = stage_config
        manifest.stage_config_sha256[stage] = digest_json(stage_config)
        if stage == "stitched_routes":
            manifest.external_assets = self._external_asset_digests()
        manifest.completed_stages.append(stage)
        manifest.current_stage = stage
        manifest.status = "running"
        manifest.failure = None
        store.save_manifest(manifest)

    @staticmethod
    def _write_trees(store: ArtifactStore, stage: str, payload: list[dict[str, Any]]) -> None:
        """Write the molecule-centric strategy trees for a stage.

        This is the form a chemist reads: one node per molecule with the reaction
        that makes it inlined, reaction SMILES written forward. The checkpoint JSON
        beside it exists to be digest-chained, not read.
        """

        trees = []
        for item in payload:
            if "route" not in item:
                trees.append(item)
                continue
            tree = op_plan_to_strategy_tree(Route.model_validate(item["route"]).to_op_plan())
            if item.get("strategy"):
                tree = {"strategy_prompt": item["strategy"]} | tree
            trees.append(tree)
        (store.root / f"{stage}.tree.json").write_text(
            json.dumps(trees, indent=2, ensure_ascii=False) + "\n"
        )

    @staticmethod
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

    async def _review_one_route(
        self,
        item: dict[str, Any],
        constraint: str | None,
        context: AgentContext,
    ) -> dict[str, Any]:
        """Run the Critic-Editor loop for one built route.

        Extracted so the caller's per-item guard wraps a single call. A route that
        cannot be reviewed keeps its built form and records why, rather than taking
        the rest of the stage down with it.
        """

        route = Route.model_validate(item["route"])
        iterations: list[dict[str, Any]] = []
        snapshots = {0: route}
        critique = None
        try:
            if self.config.agents.edit_iteration_limit > 0:
                critique = await self.components.critic.run(
                    CriticRequest(route=route, constraint=constraint), context
                )
                for iteration in range(1, self.config.agents.edit_iteration_limit + 1):
                    if critique.is_resolved:
                        break
                    edited = await self.components.editor.run(
                        EditorRequest(
                            route=route,
                            critique=critique,
                            strategy=item["strategy"],
                            iteration=iteration,
                            max_iterations=self.config.agents.edit_iteration_limit,
                            max_turns=self.config.agents.editor_turn_limit,
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
                    critique = await self.components.critic.run(
                        CriticRequest(route=route, constraint=constraint), context
                    )
        except Exception as exc:
            return {
                "status": "review_failed",
                "strategy": item["strategy"],
                "route": route.model_dump(mode="json"),
                "final_critique": None,
                "iterations": iterations,
                "failure": {
                    "category": type(exc).__name__,
                    "message": redact_sensitive_text(str(exc)),
                },
            }
        return {
            "status": "ok",
            "strategy": item["strategy"],
            "route": route.model_dump(mode="json"),
            "final_critique": (critique.model_dump(mode="json") if critique is not None else None),
            "iterations": iterations,
        }

    async def run(
        self,
        target: str,
        output: str | Path,
        *,
        target_id: str | None = None,
        constraint: str | None = None,
        resume: bool = False,
        stages: tuple[str, ...] | None = None,
        seed_strategies: list[str] | None = None,
    ) -> RunManifest:
        requested_stages = tuple(stages or STAGES)
        unknown = sorted(set(requested_stages) - set(STAGES))
        if unknown:
            raise ValueError(f"unknown pipeline stage(s): {', '.join(unknown)}")
        if not requested_stages:
            raise ValueError("at least one pipeline stage must be requested")
        if len(requested_stages) != len(set(requested_stages)):
            raise ValueError("pipeline stages must not contain duplicates")
        if seed_strategies is not None and "strategies" not in requested_stages:
            raise ValueError("provided strategies require the strategies stage")

        store = ArtifactStore(output)
        expected = self._new_manifest(target, target_id, constraint)
        if store.manifest_path.exists():
            if not resume:
                raise ValueError(f"run directory {store.root} already contains a manifest")
            manifest = store.validate()
            if manifest.input_sha256 != expected.input_sha256:
                raise ValueError("cannot resume: target/input digest changed")
            manifest.status = "running"
            manifest.failure = None
            store.save_manifest(manifest)
        else:
            manifest = expected
            store.save_manifest(manifest)

        recorder = _UsageRecorder(self.components.llm, manifest.llm_usage)
        context = AgentContext(llm=recorder, run_id=manifest.run_id)
        active_stage = "created"
        stage_started: float | None = None
        try:
            available = set(manifest.completed_stages)
            for stage in STAGES:
                if stage not in requested_stages:
                    continue
                missing = [
                    dependency
                    for dependency in STAGE_DEPENDENCIES[stage]
                    if dependency not in available
                ]
                if missing:
                    raise ValueError(
                        f"stage {stage!r} requires completed stage(s): {', '.join(missing)}"
                    )
                available.add(stage)

            if "strategies" in requested_stages and "strategies" not in manifest.completed_stages:
                active_stage = "strategies"
                stage_started = time.perf_counter()
                if seed_strategies is None:
                    result = await self.components.strategy_generator.run(
                        StrategyRequest(
                            target=target,
                            count=self.config.agents.strategy_count,
                            constraint=constraint,
                        ),
                        context,
                    )
                    origin = "generated"
                else:
                    result = StrategyResult(strategies=seed_strategies)
                    origin = "provided"
                payload = result.model_dump(mode="json") | {"origin": origin}
                self._checkpoint(store, manifest, active_stage, payload)
                self._record_runtime(store, manifest, recorder, active_stage, stage_started)

            if "routes" in requested_stages:
                strategies = store.read_checkpoint(manifest.artifacts["strategies"])["strategies"]

            if "routes" in requested_stages and "routes" not in manifest.completed_stages:
                active_stage = "routes"
                stage_started = time.perf_counter()
                routes = []
                for strategy in strategies:
                    # One strategy exhausting its proposals must not discard the
                    # routes already built for the others.
                    try:
                        result = await self.components.route_builder.run(
                            RouteBuildRequest(
                                target=target,
                                strategy=strategy,
                                max_steps=self.config.agents.route_step_limit,
                                proposal_attempts=self.config.agents.route_proposal_attempts,
                            ),
                            context,
                        )
                    except Exception as exc:
                        routes.append(_failure(strategy, exc))
                        continue
                    routes.append(
                        {
                            "status": "ok",
                            "strategy": strategy,
                            "termination": result.termination,
                            "route": result.route.model_dump(mode="json"),
                        }
                    )
                _require_one_success(routes, active_stage)
                self._checkpoint(store, manifest, active_stage, routes)
                self._write_trees(store, active_stage, routes)
                self._record_runtime(store, manifest, recorder, active_stage, stage_started)

            if "reviewed_routes" in requested_stages:
                routes_payload = store.read_checkpoint(manifest.artifacts["routes"])

            if (
                "reviewed_routes" in requested_stages
                and "reviewed_routes" not in manifest.completed_stages
            ):
                active_stage = "reviewed_routes"
                stage_started = time.perf_counter()
                reviewed = []
                for item in routes_payload:
                    if not _succeeded(item):
                        reviewed.append(item)
                        continue
                    reviewed.append(await self._review_one_route(item, constraint, context))
                _require_one_success(reviewed, active_stage)
                self._checkpoint(store, manifest, active_stage, reviewed)
                self._write_trees(store, active_stage, reviewed)
                self._record_runtime(store, manifest, recorder, active_stage, stage_started)

            if "analyses" in requested_stages or "stitched_routes" in requested_stages:
                reviewed_payload = store.read_checkpoint(manifest.artifacts["reviewed_routes"])

            if "analyses" in requested_stages and "analyses" not in manifest.completed_stages:
                active_stage = "analyses"
                stage_started = time.perf_counter()
                analyses = []
                for item in reviewed_payload:
                    if "route" not in item:
                        analyses.append(item)
                        continue
                    try:
                        analysis = await self.components.analyst.run(
                            AnalystRequest(
                                route=Route.model_validate(item["route"]),
                                constraint=constraint,
                            ),
                            context,
                        )
                    except Exception as exc:
                        analyses.append(_failure(item.get("strategy", ""), exc))
                        continue
                    analyses.append(analysis.model_dump(mode="json") | {"status": "ok"})
                _require_one_success(analyses, active_stage)
                self._checkpoint(store, manifest, active_stage, analyses)
                self._record_runtime(store, manifest, recorder, active_stage, stage_started)

            if (
                "stitched_routes" in requested_stages
                and "stitched_routes" not in manifest.completed_stages
            ):
                active_stage = "stitched_routes"
                stage_started = time.perf_counter()
                stitched = []
                if self.config.completion.enabled:
                    if self.components.leaf_completer is None:
                        raise ValueError("completion is enabled but no leaf completer was injected")
                    for item in reviewed_payload:
                        if "route" not in item:
                            stitched.append(item)
                            continue
                        try:
                            result = complete_route(
                                Route.model_validate(item["route"]),
                                self.components.leaf_completer,
                            )
                        except Exception as exc:
                            stitched.append(_failure(item.get("strategy", ""), exc))
                            continue
                        stitched.append(result.model_dump(mode="json") | {"status": "ok"})
                    _require_one_success(stitched, active_stage)
                self._checkpoint(store, manifest, active_stage, stitched)
                self._record_runtime(store, manifest, recorder, active_stage, stage_started)

            is_complete = all(stage in manifest.completed_stages for stage in STAGES)
            manifest.status = "complete" if is_complete else "partial"
            manifest.current_stage = "complete" if is_complete else manifest.completed_stages[-1]
            manifest.llm_usage = recorder.snapshot()
            store.save_manifest(manifest)
            return manifest
        except Exception as exc:
            if stage_started is not None and active_stage not in manifest.stage_durations_seconds:
                manifest.stage_durations_seconds[active_stage] = round(
                    time.perf_counter() - stage_started, 6
                )
            manifest.status = "failed"
            manifest.current_stage = active_stage
            manifest.failure = Failure(
                stage=active_stage,
                category=type(exc).__name__,
                message=redact_sensitive_text(str(exc)),
            )
            manifest.llm_usage = recorder.snapshot()
            store.save_manifest(manifest)
            raise


__all__ = ["PipelineComponents", "STAGES", "STAGE_DEPENDENCIES", "SynthexPipeline"]
