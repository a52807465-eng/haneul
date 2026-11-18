"""Versioned models for the ReactionJSON graph-edit representation."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator
from rdkit import Chem


def validate_mapped_smiles(value: str) -> str:
    """Validate a parseable SMILES with one unique positive map per atom."""

    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        raise ValueError("must be a parseable SMILES string")
    maps = [atom.GetAtomMapNum() for atom in molecule.GetAtoms()]
    if any(map_number <= 0 for map_number in maps):
        raise ValueError("every atom must have a positive atom-map number")
    if len(maps) != len(set(maps)):
        raise ValueError("atom-map numbers must be unique")
    return value


class _Operation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BreakBond(_Operation):
    op: Literal["break_bond"]
    map_a: int = Field(gt=0)
    map_b: int = Field(gt=0)


class AddBond(_Operation):
    op: Literal["add_bond"]
    map_a: int = Field(gt=0)
    map_b: int = Field(gt=0)
    order: Literal[1, 2, 3] = 1


class ChangeBondOrder(_Operation):
    op: Literal["change_bond_order"]
    map_a: int = Field(gt=0)
    map_b: int = Field(gt=0)
    delta: Literal[-2, -1, 1, 2]


class ChangeAtom(_Operation):
    op: Literal["change_atom"]
    map_idx: int = Field(gt=0)
    symbol: str = Field(min_length=1)


class SetExplicitH(_Operation):
    op: Literal["set_explicit_h"]
    map_idx: int = Field(gt=0)
    n: int = Field(ge=0)


class AddGroup(_Operation):
    op: Literal["add_group"]
    map_idx: int = Field(gt=0)
    fragment_smiles: str = Field(min_length=1)
    order: Literal[1, 2, 3] = 1


class RemoveGroup(_Operation):
    op: Literal["remove_group"]
    map_idx: int = Field(gt=0)
    map_root_idx: int = Field(gt=0)


class InvertStereocenter(_Operation):
    op: Literal["invert_stereocenter"]
    map_idx: int = Field(gt=0)


class ClearStereocenter(_Operation):
    op: Literal["clear_stereocenter"]
    map_idx: int = Field(gt=0)


class SetBondStereo(_Operation):
    op: Literal["set_bond_stereo"]
    map_a: int = Field(gt=0)
    map_b: int = Field(gt=0)
    stereo: Literal["E", "Z"]
    map_ref_a: int | None = Field(default=None, gt=0)
    map_ref_b: int | None = Field(default=None, gt=0)


Operation: TypeAlias = Annotated[
    BreakBond
    | AddBond
    | ChangeBondOrder
    | ChangeAtom
    | SetExplicitH
    | AddGroup
    | RemoveGroup
    | InvertStereocenter
    | ClearStereocenter
    | SetBondStereo,
    Field(discriminator="op"),
]


class Reaction(BaseModel):
    """A deterministic retrosynthetic disconnection in ReactionJSON 1.0."""

    model_config = ConfigDict(extra="forbid")

    format_version: Literal["reactionjson-1.0"] = "reactionjson-1.0"
    product: str = Field(min_length=1, description="Atom-mapped product SMILES")
    operations: list[Operation] = Field(min_length=1)
    description: str | None = None
    rationale: str | None = None
    conditions: str | None = None

    @field_validator("product")
    @classmethod
    def _mapped_product(cls, value: str) -> str:
        return validate_mapped_smiles(value)


def reaction_schema() -> dict:
    """Return the authoritative JSON Schema for ReactionJSON 1.0."""

    return Reaction.model_json_schema()


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
    "RemoveGroup",
    "SetBondStereo",
    "SetExplicitH",
    "reaction_schema",
    "validate_mapped_smiles",
]
