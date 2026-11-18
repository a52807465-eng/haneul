"""Analyst agent producing the manuscript's multi-dimensional assessment."""

from __future__ import annotations

import json
import re
from typing import Any

from synthex.agents._parse import parse_json
from synthex.agents.base import AgentContext
from synthex.agents.models import AnalysisResult, AnalystRequest
from synthex.llm import LLMRequest, Message, ThinkingLevel
from synthex.prompts.analyst_v1 import route_analysis_prompt
from synthex.routejson import op_plan_to_tree

TAGS = (
    "route_summary",
    "key_steps",
    "stereoselectivity",
    "handover_point",
    "convergence",
    "redundant_reactions",
    "ops_completeness",
    "reaction_conditions",
    "starting_materials",
    "feasibility_verdict",
    "asymmetric_catalysis",
    "cascade_reactions",
    "route_score",
)


def _route_text(tree: dict[str, Any]) -> str:
    lines: list[str] = []
    mol_index = 0
    step_index = 0

    def label(index: int) -> str:
        output = ""
        while True:
            output = chr(ord("A") + index % 26) + output
            index = index // 26 - 1
            if index < 0:
                return output

    def visit(node: dict[str, Any], depth: int) -> None:
        nonlocal mol_index, step_index
        prefix = "  " * depth
        current_label = label(mol_index)
        mol_index += 1
        children = node.get("children") or []
        if not children:
            lines.append(f"{prefix}[Mol {current_label}] BUILDING BLOCK: {node.get('smiles')}")
            return
        reaction = children[0]
        metadata = reaction.get("metadata") or {}
        lines.append(
            f"{prefix}[Mol {current_label}] [Step {step_index}] MOLECULE: {node.get('smiles')}"
        )
        lines.append(f"{prefix}  reaction_smiles: {reaction.get('smiles', '')}")
        if metadata.get("operations"):
            lines.append(f"{prefix}  ops: {json.dumps(metadata['operations'])}")
        if metadata.get("reaction_prompt"):
            lines.append(f"{prefix}  strategy: {metadata['reaction_prompt']}")
        step_index += 1
        for reactant in reaction.get("children") or []:
            visit(reactant, depth + 1)

    visit(tree, 0)
    return "\n".join(lines)


def _parse_analysis(text: str) -> AnalysisResult:
    try:
        return AnalysisResult.model_validate(parse_json(text))
    except (ValueError, TypeError):
        pass
    details: dict[str, Any] = {}
    for tag in TAGS:
        match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
        if not match:
            raise ValueError(f"Analyst response is missing <{tag}>")
        details[tag] = parse_json(match.group(1))
    verdict = details["feasibility_verdict"] or {}
    route_summary = details["route_summary"] or {}
    key_steps = [
        str(item.get("step_description") or f"Step {item.get('step_index')}")
        for item in (details["key_steps"] or {}).get("key_steps", [])
    ]
    risks = [str(item.get("risk", "")) for item in verdict.get("top_risks", [])]
    return AnalysisResult(
        feasibility_category=str(verdict.get("overall_rating", "unknown")),
        summary=str(
            verdict.get("one_line_summary") or route_summary.get("summary") or "No summary returned"
        ),
        key_steps=key_steps,
        risks=risks,
        details=details,
    )


class Analyst:
    name = "analyst"
    prompt_version = "analyst-v1"

    def __init__(
        self,
        temperature: float = 0.3,
        thinking_level: ThinkingLevel | None = None,
        attempts: int = 3,
    ) -> None:
        self.temperature = temperature
        self.thinking_level = thinking_level
        #: _parse_analysis requires all thirteen tags when the JSON form is absent,
        #: so a single missing tag used to fail the analyses stage outright.
        self.attempts = max(1, attempts)

    async def run(self, request: AnalystRequest, context: AgentContext) -> AnalysisResult:
        tree = op_plan_to_tree(request.route.to_op_plan())
        prompt = (
            route_analysis_prompt.replace("{{TARGET_MOLECULE}}", request.route.target)
            .replace(
                "{{USER_CONSTRAINT}}", request.constraint or "No specific constraint provided."
            )
            .replace("{{ROUTE_TREE}}", _route_text(tree))
        )
        messages: list[Message] = [Message(role="user", content=prompt)]
        last_error: Exception | None = None
        for _ in range(self.attempts):
            response = await context.llm.generate(
                LLMRequest(
                    system="You are the SynthEx route Analyst.",
                    messages=tuple(messages),
                    temperature=self.temperature,
                    thinking_level=self.thinking_level,
                )
            )
            try:
                result = _parse_analysis(response.text)
            except (ValueError, TypeError) as exc:
                last_error = exc
                messages = [
                    Message(role="user", content=prompt),
                    Message(role="assistant", content=response.text),
                    Message(
                        role="user",
                        content=(
                            f"That reply could not be read: {exc} Return every required "
                            "tag, each containing valid JSON, with no commentary."
                        ),
                    ),
                ]
                continue
            context.emit({"agent": self.name, "prompt_version": self.prompt_version})
            return result
        raise ValueError(
            f"Analyst returned no readable analysis in {self.attempts} attempts: {last_error}"
        )


__all__ = ["Analyst"]
