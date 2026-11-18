from __future__ import annotations

import asyncio
import json

import pytest

from synthex.agents import Analyst, Critic, Editor, RouteBuilder, StrategyGenerator
from synthex.artifacts import ArtifactStore, digest_file
from synthex.config import ManuscriptConfig
from synthex.llm import ReplayLLMClient
from synthex.pipeline import PipelineComponents, SynthexPipeline


def _pipeline(responses, *, model="gemini-3.1-pro-preview", edit_rounds=1):
    client = ReplayLLMClient(responses)
    config = ManuscriptConfig(
        llm={"model": model},
        agents={
            "strategy_count": 1,
            "route_step_limit": 3,
            "edit_iteration_limit": edit_rounds,
        },
        completion={"enabled": False},
    )
    components = PipelineComponents(
        llm=client,
        strategy_generator=StrategyGenerator(),
        route_builder=RouteBuilder(),
        critic=Critic(),
        editor=Editor(),
        analyst=Analyst(),
    )
    return SynthexPipeline(config, components), client


def _responses():
    return [
        {
            "text": '{"strategies": ["disconnect centrally"]}',
            "usage": {
                "input_tokens": 10,
                "cached_input_tokens": 3,
                "output_tokens": 5,
                "thought_tokens": 2,
                "total_tokens": 17,
            },
        },
        {
            "text": json.dumps(
                {
                    "done": False,
                    "step": {
                        "id": "S1",
                        "description": "central disconnection",
                        "operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                        "kind": "key",
                    },
                }
            )
        },
        {"text": '{"done": true}'},
        {
            "text": json.dumps(
                {
                    "is_satisfactory": True,
                    "overall_feedback": "acceptable",
                    "problematic_steps": [],
                }
            )
        },
        {
            "text": json.dumps(
                {
                    "feasibility_category": "acceptable",
                    "summary": "Plausible",
                    "key_steps": ["central disconnection"],
                    "risks": [],
                    "details": {},
                }
            )
        },
    ]


def test_pipeline_writes_digest_linked_checkpoints_and_resumes(tmp_path):
    pipeline, client = _pipeline(_responses())
    manifest = asyncio.run(pipeline.run("CCCC", tmp_path / "run", target_id="T1"))

    assert manifest.status == "complete"
    assert manifest.completed_stages == [
        "strategies",
        "routes",
        "reviewed_routes",
        "analyses",
        "stitched_routes",
    ]
    assert manifest.artifacts["routes"].parent_sha256 == manifest.artifacts["strategies"].sha256
    assert ArtifactStore(tmp_path / "run").validate().status == "complete"
    assert len(client.requests) == 5
    assert manifest.llm_usage == {
        "calls": 5,
        "input_tokens": 10,
        "cached_input_tokens": 3,
        "output_tokens": 5,
        "thought_tokens": 2,
        "total_tokens": 17,
    }

    resumed, replay = _pipeline([])
    again = asyncio.run(resumed.run("CCCC", tmp_path / "run", target_id="T1", resume=True))
    assert again.status == "complete"
    assert replay.requests == []


def test_pipeline_runs_stages_separately_and_records_each_stage_config(tmp_path):
    responses = _responses()
    run_dir = tmp_path / "run"

    strategies, strategy_client = _pipeline(responses[:1])
    first = asyncio.run(strategies.run("CCCC", run_dir, target_id="T1", stages=("strategies",)))
    assert first.status == "partial"
    assert first.completed_stages == ["strategies"]
    assert len(strategy_client.requests) == 1

    routes, route_client = _pipeline(responses[1:4], model="gemini-3.7-flash", edit_rounds=1)
    second = asyncio.run(
        routes.run(
            "CCCC",
            run_dir,
            target_id="T1",
            resume=True,
            stages=("routes", "reviewed_routes"),
        )
    )
    assert second.status == "partial"
    assert len(route_client.requests) == 3
    assert second.stage_configs["strategies"]["llm"]["model"] == "gemini-3.1-pro-preview"
    assert second.stage_configs["routes"]["llm"]["model"] == "gemini-3.7-flash"
    assert second.stage_configs["reviewed_routes"]["agents"]["edit_iteration_limit"] == 1

    analyses, analysis_client = _pipeline(responses[4:])
    third = asyncio.run(
        analyses.run("CCCC", run_dir, target_id="T1", resume=True, stages=("analyses",))
    )
    assert third.status == "partial"
    assert len(analysis_client.requests) == 1

    completion, completion_client = _pipeline([])
    final = asyncio.run(
        completion.run(
            "CCCC",
            run_dir,
            target_id="T1",
            resume=True,
            stages=("stitched_routes",),
        )
    )
    assert final.status == "complete"
    assert completion_client.requests == []
    assert ArtifactStore(run_dir).validate().status == "complete"


def test_zero_edit_rounds_skips_critic_and_editor(tmp_path):
    pipeline, client = _pipeline(_responses()[:3], edit_rounds=0)
    run_dir = tmp_path / "run"
    manifest = asyncio.run(
        pipeline.run(
            "CCCC",
            run_dir,
            stages=("strategies", "routes", "reviewed_routes"),
        )
    )

    assert manifest.status == "partial"
    assert len(client.requests) == 3
    reviewed = json.loads((run_dir / "reviewed_routes.json").read_text())
    assert reviewed[0]["final_critique"] is None
    assert reviewed[0]["iterations"] == []


def test_stage_dependencies_are_enforced_before_a_model_call(tmp_path):
    pipeline, client = _pipeline([])
    with pytest.raises(ValueError, match="requires completed stage"):
        asyncio.run(pipeline.run("CCCC", tmp_path / "run", stages=("routes",)))
    assert client.requests == []


def test_provided_strategy_is_checkpointed_without_an_llm_call(tmp_path):
    pipeline, client = _pipeline([])
    run_dir = tmp_path / "run"
    manifest = asyncio.run(
        pipeline.run(
            "CCCC",
            run_dir,
            stages=("strategies",),
            seed_strategies=["use the shared benchmark disconnection"],
        )
    )

    payload = json.loads((run_dir / "strategies.json").read_text())
    assert payload == {
        "origin": "provided",
        "strategies": ["use the shared benchmark disconnection"],
    }
    assert manifest.llm_usage["calls"] == 0
    assert client.requests == []


def test_run_validation_rejects_a_broken_checkpoint_chain(tmp_path):
    pipeline, _ = _pipeline(_responses())
    asyncio.run(pipeline.run("CCCC", tmp_path / "run", target_id="T1"))
    store = ArtifactStore(tmp_path / "run")
    manifest = store.load_manifest()
    manifest.artifacts["routes"].parent_sha256 = "0" * 64
    store.save_manifest(manifest)

    with pytest.raises(ValueError, match="parent digest mismatch"):
        store.validate()


def test_pipeline_records_structured_failure(tmp_path):
    pipeline, _ = _pipeline([])
    try:
        asyncio.run(pipeline.run("CCCC", tmp_path / "run"))
    except RuntimeError:
        pass
    manifest = ArtifactStore(tmp_path / "run").load_manifest()
    assert manifest.status == "failed"
    assert manifest.failure is not None
    assert manifest.failure.stage == "strategies"
    assert manifest.stage_durations_seconds["strategies"] >= 0


def test_pipeline_redacts_credentials_from_failure_manifest(tmp_path, monkeypatch):
    secret = "test-gemini-key-that-must-never-be-persisted"
    monkeypatch.setenv("GEMINI_API_KEY", secret)

    class FailingClient:
        async def generate(self, _request):
            raise RuntimeError(f"provider request failed: key={secret}")

    config = ManuscriptConfig(completion={"enabled": False})
    components = PipelineComponents(
        llm=FailingClient(),
        strategy_generator=StrategyGenerator(),
        route_builder=RouteBuilder(),
        critic=Critic(),
        editor=Editor(),
        analyst=Analyst(),
    )
    pipeline = SynthexPipeline(config, components)
    with pytest.raises(RuntimeError):
        asyncio.run(pipeline.run("CCCC", tmp_path / "run"))

    text = (tmp_path / "run" / "manifest.json").read_text()
    assert secret not in text
    assert "[REDACTED]" in text


def _failing_at(index, responses):
    """A client that raises on the Nth call and replays otherwise."""

    class Client:
        def __init__(self):
            self.calls = 0
            self.inner = ReplayLLMClient(responses)

        async def generate(self, request):
            self.calls += 1
            if self.calls == index:
                raise RuntimeError("provider exploded")
            return await self.inner.generate(request)

    return Client()


def test_one_failing_strategy_does_not_discard_the_others(tmp_path):
    """A route that cannot be built must not take the whole stage down with it."""
    step = json.dumps(
        {
            "done": False,
            "step": {
                "description": "central",
                "operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
            },
        }
    )
    # three strategies; the middle one's first proposal raises, and its retries
    # exhaust because the client keeps raising for that call index
    responses = [{"text": step}, {"text": '{"done": true}'}] * 3
    config = ManuscriptConfig(
        agents={"strategy_count": 3, "route_step_limit": 2, "route_proposal_attempts": 1},
        completion={"enabled": False},
    )
    client = _failing_at(3, responses)
    pipeline = SynthexPipeline(
        config,
        PipelineComponents(
            llm=client,
            strategy_generator=StrategyGenerator(),
            route_builder=RouteBuilder(),
            critic=Critic(),
            editor=Editor(),
            analyst=Analyst(),
        ),
    )
    run_dir = tmp_path / "run"
    asyncio.run(
        pipeline.run(
            "CCCC",
            run_dir,
            stages=("strategies", "routes"),
            seed_strategies=["a", "b", "c"],
        )
    )

    routes = json.loads((run_dir / "routes.json").read_text())
    assert len(routes) == 3
    statuses = [item.get("status") for item in routes]
    assert statuses.count("ok") == 2, statuses
    assert statuses.count("failed") == 1
    failed = next(item for item in routes if item["status"] == "failed")
    # a provider error is not a validation error, so it is recorded as itself
    assert failed["failure"]["category"] == "RuntimeError"
    assert "strategy" in failed


def test_a_stage_still_fails_when_every_item_fails(tmp_path):
    """Partial success is recorded; total failure is an error."""

    class AlwaysFails:
        async def generate(self, _request):
            raise RuntimeError("provider down")

    config = ManuscriptConfig(
        agents={"strategy_count": 2, "route_step_limit": 1, "route_proposal_attempts": 1},
        completion={"enabled": False},
    )
    pipeline = SynthexPipeline(
        config,
        PipelineComponents(
            llm=AlwaysFails(),
            strategy_generator=StrategyGenerator(),
            route_builder=RouteBuilder(),
            critic=Critic(),
            editor=Editor(),
            analyst=Analyst(),
        ),
    )
    run_dir = tmp_path / "run"
    with pytest.raises(ValueError, match="every item in stage 'routes' failed"):
        asyncio.run(
            pipeline.run(
                "CCCC", run_dir, stages=("strategies", "routes"), seed_strategies=["a", "b"]
            )
        )
    assert not (run_dir / "routes.json").exists()


def test_a_failed_route_is_carried_through_the_later_stages(tmp_path):
    """Downstream stages must skip a failed item rather than crash on it."""
    step = json.dumps(
        {
            "done": False,
            "step": {
                "description": "central",
                "operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
            },
        }
    )
    critique = json.dumps(
        {"is_satisfactory": True, "overall_feedback": "ok", "problematic_steps": []}
    )
    analysis = json.dumps(
        {
            "feasibility_category": "acceptable",
            "summary": "ok",
            "key_steps": [],
            "risks": [],
            "details": {},
        }
    )
    responses = [
        {"text": step},
        {"text": '{"done": true}'},
        {"text": step},
        {"text": '{"done": true}'},
        {"text": critique},
        {"text": analysis},
    ]
    config = ManuscriptConfig(
        agents={
            "strategy_count": 2,
            "route_step_limit": 2,
            "route_proposal_attempts": 1,
            "edit_iteration_limit": 1,
        },
        completion={"enabled": False},
    )
    client = _failing_at(1, responses)  # first strategy's route build fails
    pipeline = SynthexPipeline(
        config,
        PipelineComponents(
            llm=client,
            strategy_generator=StrategyGenerator(),
            route_builder=RouteBuilder(),
            critic=Critic(),
            editor=Editor(),
            analyst=Analyst(),
        ),
    )
    run_dir = tmp_path / "run"
    manifest = asyncio.run(pipeline.run("CCCC", run_dir, seed_strategies=["a", "b"]))

    assert manifest.status == "complete"
    reviewed = json.loads((run_dir / "reviewed_routes.json").read_text())
    analyses = json.loads((run_dir / "analyses.json").read_text())
    assert [item.get("status") for item in reviewed] == ["failed", "ok"]
    assert [item.get("status") for item in analyses] == ["failed", "ok"]
    ArtifactStore(run_dir).validate()


def test_checkpoint_bytes_match_recorded_digest_and_tampering_is_detected(tmp_path):
    """Checkpoints must be byte-stable across platforms without losing validation."""
    store = ArtifactStore(tmp_path / "store")
    payload = {"strategies": ["가능한 한 짧고 상업적으로 구입 가능한 출발 물질"], "origin": "generated"}

    reference = store.write_checkpoint("strategies", payload)
    raw = (store.root / reference.path).read_bytes()

    assert b"\r\n" not in raw
    assert reference.sha256 == digest_file(store.root / reference.path)
    assert store.read_checkpoint(reference) == payload

    (store.root / reference.path).write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="checkpoint digest mismatch"):
        store.read_checkpoint(reference)
