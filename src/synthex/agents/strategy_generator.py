"""Strategy Generator agent."""

from __future__ import annotations

from synthex.agents._parse import parse_json
from synthex.agents.base import AgentContext
from synthex.agents.models import StrategyRequest, StrategyResult
from synthex.llm import LLMRequest, Message, ThinkingLevel
from synthex.prompts.strategy_generator_v1 import (
    stock_constraint_section,
    stock_constraint_sentence,
    strategy_generator_prompt,
)


class StrategyGenerator:
    name = "strategy_generator"
    prompt_version = "strategy-generator-v1"

    def __init__(
        self, temperature: float = 0.1, thinking_level: ThinkingLevel | None = None
    ) -> None:
        self.temperature = temperature
        self.thinking_level = thinking_level

    async def run(self, request: StrategyRequest, context: AgentContext) -> StrategyResult:
        example_items = ",\n    ".join(
            f'"Strategy {index} text ..."' for index in range(1, request.count + 1)
        )
        stock_section = ""
        constraint_sentence = ""
        if request.stock_summary:
            stock_section = stock_constraint_section.replace(
                "{{STOCK_SUMMARY}}", request.stock_summary
            )
            constraint_sentence = stock_constraint_sentence
        system = (
            strategy_generator_prompt.replace("{{STOCK_SECTION}}", stock_section)
            .replace("{{STOCK_CONSTRAINT_SENTENCE}}", constraint_sentence)
            .replace("{{N_STRATEGIES}}", str(request.count))
            .replace("{{EXAMPLE_ITEMS}}", example_items)
        )
        user = f"TARGET MOLECULE SMILES: {request.target}"
        if request.starting_material:
            user += f"\nREQUIRED STARTING MATERIAL: {request.starting_material}"
        if request.constraint:
            user += f"\nADDITIONAL GUIDANCE: {request.constraint}"
        response = await context.llm.generate(
            LLMRequest(
                system=system,
                messages=(Message(role="user", content=user),),
                temperature=self.temperature,
                thinking_level=self.thinking_level,
                response_schema=StrategyResult.model_json_schema(),
            )
        )
        result = StrategyResult.model_validate(parse_json(response.text))
        if len(result.strategies) > request.count:
            result = StrategyResult(strategies=result.strategies[: request.count])
        context.emit({"agent": self.name, "prompt_version": self.prompt_version})
        return result


__all__ = ["StrategyGenerator"]
