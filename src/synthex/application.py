"""Composition root for live manuscript runs."""

from __future__ import annotations

from synthex.agents import Analyst, Critic, Editor, RouteBuilder, StrategyGenerator
from synthex.config import ManuscriptConfig, validate_external_assets
from synthex.integrations.gemini import GeminiClient
from synthex.pipeline import PipelineComponents, SynthexPipeline


def create_pipeline(
    config: ManuscriptConfig, *, include_completion: bool = True
) -> SynthexPipeline:
    """Initialize the agents and only initialize AiZynthFinder when requested."""

    if include_completion:
        validate_external_assets(config)
    llm = GeminiClient(
        model=config.llm.model,
        use_vertex_ai=config.llm.use_vertex_ai,
        timeout_seconds=config.llm.timeout_seconds,
        max_attempts=config.llm.max_attempts,
        thinking_level=config.llm.thinking_level,
    )
    leaf_completer = None
    if include_completion and config.completion.enabled:
        from synthex.integrations.aizynthfinder import AiZynthFinderLeafCompleter

        assert config.completion.aizynthfinder_config is not None
        leaf_completer = AiZynthFinderLeafCompleter(
            config.completion.aizynthfinder_config,
            budget=config.completion.budget,
            stocks=config.completion.stocks,
            expansion_policies=config.completion.expansion_policies,
            filter_policies=config.completion.filter_policies,
        )
    return SynthexPipeline(
        config,
        PipelineComponents(
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
            leaf_completer=leaf_completer,
        ),
    )


__all__ = ["create_pipeline"]
