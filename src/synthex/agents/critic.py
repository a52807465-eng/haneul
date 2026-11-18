"""Critic agent for structured route findings."""

from __future__ import annotations

import json
import re

from synthex.agents._feedback import describe_validation_error
from synthex.agents._parse import parse_json
from synthex.agents.base import AgentContext
from synthex.agents.models import CriticRequest, Critique
from synthex.llm import LLMRequest, Message, ThinkingLevel
from synthex.prompts.critic_v1 import validator_prompt
from synthex.routejson import atom_stereo_accounting, render_plan


def _accounting_block(rendered: dict) -> str:
    """Per-step atom and stereo accounting for the rendered route.

    ``render_plan`` already emits one ``product>>precursors`` entry per step, so
    the balance is computed from those rather than re-executing the operations.
    """

    entries = rendered.get("route") or []
    if not entries:
        return "No steps to account for."
    blocks = []
    for entry in entries:
        reaction = entry.get("smiles") or ""
        product, _, precursors = reaction.partition(">>")
        blocks.append(
            f"Step {entry.get('id')}: {reaction}\n" + atom_stereo_accounting(product, precursors)
        )
    return "\n\n".join(blocks)


class Critic:
    name = "critic"
    prompt_version = "critic-v2-atom-accounting"

    def __init__(
        self,
        temperature: float = 0.1,
        thinking_level: ThinkingLevel | None = "medium",
        attempts: int = 3,
    ) -> None:
        self.temperature = temperature
        self.thinking_level = thinking_level
        #: The Route Builder retries a malformed proposal; the Critic used to have
        #: no retry at all, so one unparseable reply failed the reviewed_routes
        #: stage after the entire route-building spend was already gone.
        self.attempts = max(1, attempts)

    async def run(self, request: CriticRequest, context: AgentContext) -> Critique:
        rendered = render_plan(request.route.to_op_plan())
        prompt = (
            validator_prompt.replace("{{TARGET_MOLECULE}}", request.route.target)
            .replace("{{USER_STRATEGY}}", request.constraint or "No additional constraint")
            .replace("{{RENDERED_ROUTE}}", json.dumps(rendered, indent=2))
            .replace("{{ATOM_ACCOUNTING}}", _accounting_block(rendered))
        )
        messages: list[Message] = [Message(role="user", content=prompt)]
        last_error: Exception | None = None
        for _ in range(self.attempts):
            response = await context.llm.generate(
                LLMRequest(
                    system="You are the SynthEx route Critic.",
                    messages=tuple(messages),
                    temperature=self.temperature,
                    thinking_level=self.thinking_level,
                )
            )
            try:
                tagged = re.search(r"<feedback>\s*(.*?)\s*</feedback>", response.text, re.DOTALL)
                payload = parse_json(tagged.group(1) if tagged else response.text)
                result = Critique.model_validate(payload)
            except (ValueError, TypeError) as exc:
                last_error = exc
                messages = [
                    Message(role="user", content=prompt),
                    Message(role="assistant", content=response.text),
                    Message(
                        role="user",
                        content=(
                            "That reply could not be read: "
                            f"{describe_validation_error(exc)} Return only the "
                            "<feedback> block as valid JSON, with no commentary and no "
                            "comments inside the JSON."
                        ),
                    ),
                ]
                continue
            context.emit({"agent": self.name, "prompt_version": self.prompt_version})
            return result
        raise ValueError(
            f"Critic returned no readable feedback in {self.attempts} attempts: {last_error}"
        )


__all__ = ["Critic"]
