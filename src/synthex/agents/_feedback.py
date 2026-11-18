"""Turn a validation failure into feedback a model can act on.

The Route Builder retries a rejected proposal with ``previous_error`` set, but a
raw pydantic ValidationError is a poor instruction: it names the failing tag and
then lists every valid tag in one line, with a URL. Observed live, models retried
the same mistake until the budget was gone.

This renders the failure as a short imperative, and for the case that actually
recurs -- an operation name outside the closed vocabulary -- names the nearest
valid operation.
"""

from __future__ import annotations

from difflib import get_close_matches

from pydantic import ValidationError

from synthex.reactionjson.model import Operation

#: The closed vocabulary, read off the discriminated union so it cannot drift.
OPERATION_NAMES: tuple[str, ...] = tuple(
    sorted(
        member.model_fields["op"].annotation.__args__[0] for member in Operation.__origin__.__args__
    )
)


#: Operation name -> the argument keys it accepts, read off the same union.
OPERATION_FIELDS: dict[str, frozenset[str]] = {
    member.model_fields["op"].annotation.__args__[0]: frozenset(
        name for name in member.model_fields if name != "op"
    )
    for member in Operation.__origin__.__args__
}


def _suggest_operation(name: object, given: object) -> str | None:
    """Name the operation the caller most likely meant.

    The argument keys are a stronger signal than the name: a model that invents
    ``attach_fragment`` still passes ``map_idx`` and ``fragment_smiles``, which
    only ``add_group`` accepts. Name similarity is the fallback.
    """

    if isinstance(given, dict):
        keys = frozenset(key for key in given if key != "op")
        if keys:
            # An exact field-set match is decisive; a subset match is only useful
            # when it is unique, since {map_a, map_b} is a subset of four operations.
            for candidates in (
                [op for op, fields in OPERATION_FIELDS.items() if keys == fields],
                [op for op, fields in OPERATION_FIELDS.items() if keys <= fields],
            ):
                if len(candidates) == 1:
                    return candidates[0]
    if isinstance(name, str):
        closest = get_close_matches(name, OPERATION_NAMES, n=1, cutoff=0.3)
        if closest:
            return closest[0]
    return None


def _operation_hint(name: object, given: object = None) -> str:
    suggestion = _suggest_operation(name, given)
    tail = f" Did you mean {suggestion!r}?" if suggestion else ""
    return (
        f"{name!r} is not an operation. The vocabulary is closed: "
        f"{', '.join(OPERATION_NAMES)}.{tail}"
    )


def describe_validation_error(error: Exception) -> str:
    """Return a short, imperative description of why a proposal was rejected."""

    if not isinstance(error, ValidationError):
        return f"{type(error).__name__}: {error}"

    lines: list[str] = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"]) or "(root)"
        kind = detail.get("type", "")
        given = detail.get("input")
        if kind in {"union_tag_invalid", "union_tag_not_found"}:
            if kind == "union_tag_not_found":
                lines.append(
                    f"{location}: every operation must carry its name under the key "
                    f"'op'. Got keys {sorted(given)} instead."
                    if isinstance(given, dict)
                    else f"{location}: operation is missing the 'op' key."
                )
                continue
            name = given.get("op") if isinstance(given, dict) else given
            hint = _operation_hint(name, given)
            lines.append(f"{location}: {hint}")
            continue
        if kind == "literal_error":
            expected = detail.get("ctx", {}).get("expected", "")
            lines.append(f"{location}: {given!r} is not allowed; expected {expected}.")
            continue
        if kind == "missing":
            lines.append(f"{location}: required field is missing.")
            continue
        if kind == "extra_forbidden":
            lines.append(f"{location}: unexpected field; remove it.")
            continue
        lines.append(f"{location}: {detail.get('msg', kind)} (got {given!r}).")

    # Collapse the identical message every element of a list produces.
    unique: list[str] = []
    for line in lines:
        if line not in unique:
            unique.append(line)
    return " ".join(unique)


__all__ = ["OPERATION_NAMES", "describe_validation_error"]
