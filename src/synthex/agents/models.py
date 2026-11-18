"""Typed inputs and outputs for the five manuscript agents."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from synthex.routejson import Route


class AgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StrategyRequest(AgentModel):
    target: str
    count: int = Field(default=3, ge=1)
    constraint: str | None = None
    stock_summary: str | None = None
    starting_material: str | None = None


class StrategyResult(AgentModel):
    strategies: list[str] = Field(min_length=1)


class RouteBuildRequest(AgentModel):
    target: str
    strategy: str
    max_steps: int = Field(default=25, ge=1)
    proposal_attempts: int = Field(default=3, ge=1)


class RouteBuildResult(AgentModel):
    route: Route
    termination: Literal["model_finished", "step_limit"]


class StepFinding(AgentModel):
    step_id: str | int
    severity: Literal["blocking", "suggestion"] = "blocking"
    feedback: str


class Critique(AgentModel):
    is_satisfactory: bool = False
    overall_feedback: str | None = None
    #: The Critic's ruling on the RDKit-computed atom and stereo accounting.
    balance_check: str | None = None
    problematic_steps: list[StepFinding] = Field(default_factory=list)

    @property
    def blocking_steps(self) -> list[StepFinding]:
        return [finding for finding in self.problematic_steps if finding.severity == "blocking"]

    @property
    def is_resolved(self) -> bool:
        if not self.problematic_steps:
            return True
        return self.is_satisfactory and not self.blocking_steps


class CriticRequest(AgentModel):
    route: Route
    constraint: str | None = None


class EditorRequest(AgentModel):
    route: Route
    critique: Critique
    strategy: str
    iteration: int = Field(default=1, ge=1)
    max_iterations: int = Field(default=1, ge=1)
    max_turns: int = Field(default=60, ge=1)
    history: list[dict[str, Any]] = Field(default_factory=list)
    snapshots: dict[int, Route] = Field(default_factory=dict)


class ToolRecord(AgentModel):
    turn: int
    name: str
    arguments: dict[str, Any]
    ok: bool
    result: str


class EditorResult(AgentModel):
    route: Route
    edit_log: str
    tool_history: list[ToolRecord]


class AnalystRequest(AgentModel):
    route: Route
    constraint: str | None = None


class AnalysisResult(AgentModel):
    feasibility_category: str
    summary: str
    key_steps: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class RouteBuilderTurn(AgentModel):
    done: bool = False
    step: dict[str, Any] | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _step_required_unless_done(self) -> RouteBuilderTurn:
        if not self.done and self.step is None:
            raise ValueError("a route-builder turn must provide a step or set done=true")
        return self


__all__ = [
    "AnalysisResult",
    "AnalystRequest",
    "CriticRequest",
    "Critique",
    "EditorRequest",
    "EditorResult",
    "RouteBuildRequest",
    "RouteBuildResult",
    "RouteBuilderTurn",
    "StepFinding",
    "StrategyRequest",
    "StrategyResult",
    "ToolRecord",
]
