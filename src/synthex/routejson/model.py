"""Versioned RouteJSON document models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from synthex.reactionjson.model import Operation, validate_mapped_smiles


class RouteStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^S[1-9][0-9]*$")
    description: str = ""
    operations: list[Operation] = Field(min_length=1)
    rationale: str | None = None
    conditions: str | None = None
    kind: Literal["key", "supporting"] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Route(BaseModel):
    """A linear, editable synthesis route in RouteJSON 1.0."""

    model_config = ConfigDict(extra="forbid")

    format_version: Literal["routejson-1.0"] = "routejson-1.0"
    target: str = Field(min_length=1, description="Atom-mapped target SMILES")
    steps: list[RouteStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target")
    @classmethod
    def _mapped_target(cls, value: str) -> str:
        return validate_mapped_smiles(value)

    @model_validator(mode="after")
    def _unique_step_ids(self) -> Route:
        identifiers = [step.id for step in self.steps]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("RouteJSON step identifiers must be unique")
        return self

    def to_op_plan(self) -> dict[str, Any]:
        """Return the legacy lean-plan shape consumed by deterministic renderers."""

        steps: list[dict[str, Any]] = []
        for step in self.steps:
            item: dict[str, Any] = {
                "id": step.id,
                "desc": step.description,
                "ops": [op.model_dump(exclude_none=True) for op in step.operations],
            }
            if step.rationale is not None:
                item["step_rationale"] = step.rationale
            if step.conditions is not None:
                item["reaction_conditions"] = step.conditions
            if step.kind is not None:
                item["kind"] = step.kind
            item.update(step.metadata)
            steps.append(item)
        return {"target": self.target, "steps": steps}

    @classmethod
    def from_op_plan(cls, plan: dict[str, Any]) -> Route:
        """Validate a legacy lean plan as RouteJSON 1.0."""

        route_steps = []
        known = {"id", "desc", "ops", "step_rationale", "reaction_conditions", "kind"}
        for step in plan.get("steps", []):
            route_steps.append(
                {
                    "id": step["id"],
                    "description": step.get("desc", ""),
                    "operations": step.get("ops", []),
                    "rationale": step.get("step_rationale"),
                    "conditions": step.get("reaction_conditions"),
                    "kind": step.get("kind"),
                    "metadata": {key: value for key, value in step.items() if key not in known},
                }
            )
        return cls(target=plan["target"], steps=route_steps)


def route_schema() -> dict:
    """Return the authoritative JSON Schema for RouteJSON 1.0."""

    return Route.model_json_schema()


__all__ = ["Route", "RouteStep", "route_schema"]
