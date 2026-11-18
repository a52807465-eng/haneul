from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError
from rdkit import Chem

from synthex.reactionjson import Reaction, ReactionExecutionError, apply_reaction


def _canonical_unmapped(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule)


def test_reactionjson_import_does_not_initialize_planner_dependencies():
    code = """
import sys
import synthex.reactionjson
assert 'google.genai' not in sys.modules
assert 'aizynthfinder.aizynthfinder' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_apply_reaction_breaks_mapped_bond():
    reaction = Reaction(
        product="[CH3:1][CH2:2][CH2:3][CH3:4]",
        operations=[{"op": "break_bond", "map_a": 2, "map_b": 3}],
    )
    result = apply_reaction(reaction)
    assert _canonical_unmapped(result) == "CC.CC"


def test_unknown_operation_is_rejected_before_execution():
    with pytest.raises(ValidationError):
        Reaction(
            product="[CH3:1][CH3:2]",
            operations=[{"op": "invent_reaction", "map_a": 1}],
        )


@pytest.mark.parametrize(
    "product, message",
    [
        ("CC", "positive atom-map"),
        ("[CH3:1][CH3:1]", "must be unique"),
        ("not-smiles", "parseable SMILES"),
    ],
)
def test_product_requires_complete_unique_atom_mapping(product, message):
    with pytest.raises(ValidationError, match=message):
        Reaction(
            product=product,
            operations=[{"op": "break_bond", "map_a": 1, "map_b": 2}],
        )


def test_execution_error_has_public_exception():
    reaction = Reaction(
        product="[CH3:1][CH3:2]",
        operations=[{"op": "break_bond", "map_a": 1, "map_b": 99}],
    )
    with pytest.raises(ReactionExecutionError, match="map number 99"):
        apply_reaction(reaction)
