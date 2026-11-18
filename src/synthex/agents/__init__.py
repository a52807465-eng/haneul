"""The five agents described in the SynthEx manuscript."""

from synthex.agents.analyst import Analyst
from synthex.agents.base import Agent, AgentContext
from synthex.agents.critic import Critic
from synthex.agents.editor import Editor
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
from synthex.agents.route_builder import RouteBuilder
from synthex.agents.strategy_generator import StrategyGenerator

__all__ = [
    "Agent",
    "AgentContext",
    "AnalysisResult",
    "Analyst",
    "AnalystRequest",
    "Critic",
    "CriticRequest",
    "Critique",
    "Editor",
    "EditorRequest",
    "EditorResult",
    "RouteBuildRequest",
    "RouteBuildResult",
    "RouteBuilder",
    "StrategyGenerator",
    "StrategyRequest",
    "StrategyResult",
]
