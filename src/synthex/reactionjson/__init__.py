"""Public ReactionJSON API."""

from synthex.reactionjson.executor import execute_ops, get_mapped_smiles
from synthex.reactionjson.model import (
    AddBond,
    AddGroup,
    BreakBond,
    ChangeAtom,
    ChangeBondOrder,
    ClearStereocenter,
    InvertStereocenter,
    Operation,
    Reaction,
    RemoveGroup,
    SetBondStereo,
    SetExplicitH,
    reaction_schema,
)


class ReactionExecutionError(ValueError):
    """Raised when validated ReactionJSON cannot be executed."""


def apply_reaction(reaction: Reaction | dict) -> str:
    """Validate and execute a ReactionJSON document, returning mapped precursors."""

    parsed = reaction if isinstance(reaction, Reaction) else Reaction.model_validate(reaction)
    operations = [operation.model_dump(exclude_none=True) for operation in parsed.operations]
    result, error = execute_ops(operations, parsed.product)
    if error or result is None:
        raise ReactionExecutionError(error or "Reaction execution returned no product")
    return result


__all__ = [
    "AddBond",
    "AddGroup",
    "BreakBond",
    "ChangeAtom",
    "ChangeBondOrder",
    "ClearStereocenter",
    "InvertStereocenter",
    "Operation",
    "Reaction",
    "ReactionExecutionError",
    "RemoveGroup",
    "SetBondStereo",
    "SetExplicitH",
    "apply_reaction",
    "execute_ops",
    "get_mapped_smiles",
    "reaction_schema",
]
