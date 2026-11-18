"""Run the small, deterministic reviewer demo without credentials or network access."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from synthex.agents import Analyst, Critic, Editor, RouteBuilder, StrategyGenerator
from synthex.artifacts import ArtifactStore
from synthex.config import ManuscriptConfig
from synthex.llm import ReplayLLMClient
from synthex.pipeline import PipelineComponents, SynthexPipeline

DEMO_ROOT = Path(__file__).resolve().parent


def _read_json(name: str) -> Any:
    return json.loads((DEMO_ROOT / name).read_text())


def _pipeline(responses: list[dict[str, Any]]) -> tuple[SynthexPipeline, ReplayLLMClient]:
    client = ReplayLLMClient(responses)
    config = ManuscriptConfig(
        llm={"model": "offline-replay"},
        agents={
            "strategy_count": 1,
            "route_step_limit": 3,
            "route_proposal_attempts": 1,
            "edit_iteration_limit": 1,
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


def _summarize(
    store: ArtifactStore,
    client: ReplayLLMClient,
) -> dict[str, Any]:
    manifest = store.validate()
    strategies = store.read_checkpoint(manifest.artifacts["strategies"])
    routes = store.read_checkpoint(manifest.artifacts["routes"])
    reviewed = store.read_checkpoint(manifest.artifacts["reviewed_routes"])
    analyses = store.read_checkpoint(manifest.artifacts["analyses"])
    return {
        "target_id": manifest.target_id,
        "status": manifest.status,
        "completed_stages": manifest.completed_stages,
        "strategy_origin": strategies["origin"],
        "strategy_count": len(strategies["strategies"]),
        "route_count": len(routes),
        "route_steps": len(routes[0]["route"]["steps"]),
        "critic_resolved": reviewed[0]["final_critique"]["is_satisfactory"],
        "analysis_category": analyses[0]["feasibility_category"],
        "replayed_llm_calls": len(client.requests),
        "leaf_completion": "disabled for offline demo",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/nature-demo", help="new output directory")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"output directory already exists: {output}")

    inputs = _read_json("input.json")
    pipeline, client = _pipeline(_read_json("replay-responses.json"))
    asyncio.run(
        pipeline.run(
            inputs["smiles"],
            output,
            target_id=inputs["target_id"],
        )
    )

    store = ArtifactStore(output)
    summary = _summarize(store, client)
    expected = _read_json("expected-summary.json")
    if summary != expected:
        raise RuntimeError(
            "demo output did not match demo/expected-summary.json:\n"
            + json.dumps(summary, indent=2)
        )
    (output / "demo-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    stages = " -> ".join(summary["completed_stages"])
    print("SynthEx offline demo: PASS")
    print(f"target: {summary['target_id']} ({inputs['smiles']})")
    print(f"completed stages: {stages}")
    print(f"route: {summary['route_count']} route with {summary['route_steps']} step")
    print(f"analysis: {summary['analysis_category']}")
    print(f"simulated LLM responses replayed: {summary['replayed_llm_calls']}")
    print(f"output: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
