"""Repair model-supplied step dicts before RouteJSON validation.

``RouteBuilderTurn.step`` is an untyped object and the Editor edits plan YAML
freehand, so neither the response schema nor the tool layer constrains what comes
back. Live gemini-3.7-flash produced, on different targets: ``kind`` values of
"transform", "forward", "deprotection" and "C-O bond formation"; operations keyed
on ``type`` instead of ``op``; and step ids "S3a"/"S3b" when splitting a step.

Each was fatal and none needed to be. These helpers repair what is unambiguous
and preserve what they change, so nothing the model said is silently discarded.

Both shapes are handled: the Route Builder works on RouteStep dicts, whose
operations live under ``operations``, and the Editor works on lean op-plan dicts,
whose operations live under ``ops``.
"""

from __future__ import annotations

import re
from typing import Any

#: Accepted ``RouteStep.kind`` labels.
STEP_KINDS = ("key", "supporting")

#: Keys a model reaches for instead of ``op``.
OP_KEY_ALIASES = ("type", "name", "operation")

#: The ``RouteStep.id`` pattern.
STEP_ID = re.compile(r"^S[1-9][0-9]*$")

_OPERATION_KEYS = ("operations", "ops")


def _record(step: dict[str, Any], key: str, value: Any) -> None:
    """Preserve a value the caller is about to overwrite.

    The two shapes store metadata differently. A RouteStep dict has a real
    ``metadata`` field; a lean op-plan step has metadata flattened into the step
    itself, because ``Route.to_op_plan`` merges it in and ``from_op_plan`` collects
    every unrecognised key back out. Writing a nested dict into an op-plan step
    would round-trip as ``metadata={"metadata": {...}}``.
    """

    if "ops" in step and "operations" not in step:
        step[key] = value
        return
    metadata = step.get("metadata")
    step["metadata"] = {**(metadata if isinstance(metadata, dict) else {}), key: value}


def normalize_step_kind(step: dict[str, Any]) -> None:
    """Demote an out-of-vocabulary ``kind`` rather than rejecting the step.

    ``kind`` marks a step as strategically key or supporting. It has no effect on
    the graph edits, the rendered route or leaf completion, so failing a valid
    disconnection over the label is pure waste. The reported value is kept under
    ``metadata.reported_kind``.
    """

    reported = step.get("kind")
    if reported is not None and reported not in STEP_KINDS:
        _record(step, "reported_kind", reported)
        step["kind"] = None


def normalize_operation_keys(step: dict[str, Any]) -> None:
    """Rename an aliased operation-name key to ``op``.

    ``Operation`` is a discriminated union tagged on ``op``, so an operation that
    names itself under ``type`` fails with an opaque union_tag_not_found for every
    entry in the list. The aliases are not valid ReactionJSON keys, so there is
    nothing to lose by reinterpreting them.
    """

    for key in _OPERATION_KEYS:
        operations = step.get(key)
        if not isinstance(operations, list):
            continue
        repaired: list[Any] = []
        for operation in operations:
            if isinstance(operation, dict) and "op" not in operation:
                alias = next(
                    (name for name in OP_KEY_ALIASES if isinstance(operation.get(name), str)),
                    None,
                )
                if alias is not None:
                    operation = {"op": operation[alias]} | {
                        name: value for name, value in operation.items() if name != alias
                    }
            repaired.append(operation)
        step[key] = repaired


def normalize_step(step: dict[str, Any]) -> None:
    """Apply every per-step repair."""

    normalize_step_kind(step)
    normalize_operation_keys(step)


def renumber_step_ids(steps: list[Any]) -> None:
    """Force step ids to S1..Sn, in list order.

    The Editor legitimately splits one step into two and labels them "S3a"/"S3b",
    which RouteJSON rejects. Nothing references a step by id -- operations address
    atoms by map number -- so sequential renumbering preserves the intent and the
    order. The reported id is kept under ``metadata.reported_id``.
    """

    for position, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        expected = f"S{position}"
        reported = step.get("id")
        if reported == expected:
            continue
        if isinstance(reported, str) and reported:
            _record(step, "reported_id", reported)
        step["id"] = expected


def normalize_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Repair every step of a lean op-plan in place and return it."""

    steps = plan.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, dict):
                normalize_step(step)
        renumber_step_ids(steps)
    return plan


__all__ = [
    "OP_KEY_ALIASES",
    "STEP_ID",
    "STEP_KINDS",
    "normalize_operation_keys",
    "normalize_plan",
    "normalize_step",
    "normalize_step_kind",
    "renumber_step_ids",
]
