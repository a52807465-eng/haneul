"""Route Builder agent using deterministic ReactionJSON execution."""

from __future__ import annotations

import json

from rdkit import Chem

from synthex.agents._feedback import describe_validation_error
from synthex.agents._normalize import normalize_step
from synthex.agents._parse import parse_json
from synthex.agents.base import AgentContext
from synthex.agents.models import RouteBuilderTurn, RouteBuildRequest, RouteBuildResult
from synthex.llm import LLMRequest, Message, ThinkingLevel
from synthex.prompts.route_builder_v1 import route_builder_prompt
from synthex.reactionjson import get_mapped_smiles
from synthex.routejson import Route, RouteStep, render_plan


def _ensure_mapped(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Could not parse target SMILES: {smiles!r}")
    if all(
        atom.GetAtomMapNum() > 0
        for atom in molecule.GetAtoms()  # type: ignore[call-arg]
    ):
        return Chem.MolToSmiles(molecule)
    return get_mapped_smiles(smiles)


class RouteBuilder:
    name = "route_builder"
    prompt_version = "route-builder-reactionjson-v2"

    def __init__(
        self, temperature: float = 0.3, thinking_level: ThinkingLevel | None = "medium"
    ) -> None:
        self.temperature = temperature
        self.thinking_level = thinking_level

    async def run(self, request: RouteBuildRequest, context: AgentContext) -> RouteBuildResult:
        route = Route(target=_ensure_mapped(request.target))
        for index in range(1, request.max_steps + 1):
            rendered = render_plan(route.to_op_plan()) if route.steps else route.to_op_plan()
            previous_error: str | None = None
            for attempt in range(1, request.proposal_attempts + 1):
                user = json.dumps(
                    {
                        "strategy": request.strategy,
                        "step_number": index,
                        "maximum_steps": request.max_steps,
                        "proposal_attempt": attempt,
                        "current_route": rendered,
                        "previous_error": previous_error,
                    },
                    indent=2,
                )
                response = await context.llm.generate(
                    LLMRequest(
                        system=route_builder_prompt,
                        messages=(Message(role="user", content=user),),
                        temperature=self.temperature,
                        thinking_level=self.thinking_level,
                        response_schema=RouteBuilderTurn.model_json_schema(),
                    )
                )
                try:
                    turn = RouteBuilderTurn.model_validate(parse_json(response.text))
                    if turn.done:
                        return RouteBuildResult(route=route, termination="model_finished")
                    step_data = dict(turn.step or {})
                    step_data["id"] = f"S{index}"
                    normalize_step(step_data)
                    step = RouteStep.model_validate(step_data)
                    candidate = route.model_copy(update={"steps": [*route.steps, step]})
                    render_plan(candidate.to_op_plan())
                except ValueError as exc:
                    # A raw ValidationError lists every valid tag on one line with a
                    # URL; models retried the same mistake until the budget was gone.
                    previous_error = describe_validation_error(exc)
                    continue
                route = candidate
                break
            else:
                raise ValueError(
                    f"Route Builder exhausted {request.proposal_attempts} proposals "
                    f"for step {index}: {previous_error}"
                )
            context.emit({"agent": self.name, "prompt_version": self.prompt_version, "step": index})
        return RouteBuildResult(route=route, termination="step_limit")


__all__ = ["RouteBuilder"]
