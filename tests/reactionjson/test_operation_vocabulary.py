from __future__ import annotations

import pytest
from rdkit import Chem

from synthex.reactionjson import Reaction, apply_reaction


def _canonical(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule, isomericSmiles=True)


@pytest.mark.parametrize(
    ("product", "operation", "expected"),
    [
        (
            "[CH4:1].[CH4:2]",
            {"op": "add_bond", "map_a": 1, "map_b": 2},
            "CC",
        ),
        (
            "[CH3:1][CH3:2]",
            {"op": "break_bond", "map_a": 1, "map_b": 2},
            "C.C",
        ),
        (
            "[CH3:1][CH3:2]",
            {"op": "change_bond_order", "map_a": 1, "map_b": 2, "delta": 1},
            "C=C",
        ),
        ("[CH4:1]", {"op": "change_atom", "map_idx": 1, "symbol": "N"}, "N"),
        ("[CH4:1]", {"op": "set_explicit_h", "map_idx": 1, "n": 3}, "[CH3]"),
        (
            "[CH4:1]",
            {"op": "add_group", "map_idx": 1, "fragment_smiles": "*Cl"},
            "CCl",
        ),
        (
            "[CH3:1][Cl:2]",
            {"op": "remove_group", "map_idx": 1, "map_root_idx": 2},
            "C",
        ),
        (
            "[CH3:1][C@@H:2]([NH2:3])[C:4](=[O:5])[OH:6]",
            {"op": "invert_stereocenter", "map_idx": 2},
            "C[C@H](N)C(=O)O",
        ),
        (
            "[CH3:1][C@@H:2]([NH2:3])[C:4](=[O:5])[OH:6]",
            {"op": "clear_stereocenter", "map_idx": 2},
            "CC(N)C(=O)O",
        ),
        (
            "[CH3:1][CH:2]=[CH:3][CH3:4]",
            {"op": "set_bond_stereo", "map_a": 2, "map_b": 3, "stereo": "E"},
            "C/C=C/C",
        ),
    ],
)
def test_closed_operation_vocabulary_executes(product, operation, expected):
    output = apply_reaction(Reaction(product=product, operations=[operation]))
    assert _canonical(output) == _canonical(expected)
