"""Tests for ring closure onto atoms introduced by add_group.

The executor resolves op arguments by map number. add_group preserves explicit
map numbers written into the fragment SMILES, so a later op can bond to a
newly-introduced atom (e.g. to close a ring) by referencing its tag. These
tests cover:

- a vanilla ring closure onto a tagged new atom,
- cyclic-acetal (dioxolane) protection that keeps the carbonyl carbon's map,
- the guardrail that turns the old silent-corruption failure (referencing an
  untagged map-0 atom) into a caught error,
- the collision guardrail on targets with more than 99 atoms, where a naive
  fixed tag like ``:99`` clashes with an existing atom.
"""

from __future__ import annotations

import unittest

from rdkit import Chem, RDLogger

from synthex.reactionjson.executor import execute_ops, get_mapped_smiles

RDLogger.DisableLog("rdApp.*")


def _canon(smiles: str) -> str:
    """Canonical SMILES with all atom maps stripped."""
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, f"unparseable SMILES: {smiles!r}"
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return Chem.MolToSmiles(m)


def _map_nums(smiles: str) -> set[int]:
    m = Chem.MolFromSmiles(smiles)
    return {a.GetAtomMapNum() for a in m.GetAtoms() if a.GetAtomMapNum() > 0}


class TestRingClosureOps(unittest.TestCase):
    def test_vanilla_ring_closure_via_tagged_atom(self):
        # Butane; graft a 2-atom tail onto C1, tag its end, bond it to C4 -> a
        # six-membered ring (cyclohexane). The tag makes the new atom addressable.
        mapped = get_mapped_smiles("CCCC")  # maps 1..4
        ops = [
            {"op": "add_group", "map_idx": 1, "fragment_smiles": "*C[CH3:90]", "order": 1},
            {"op": "add_bond", "map_a": 4, "map_b": 90, "order": 1},
        ]
        res, err = execute_ops(ops, mapped)
        self.assertIsNone(err)
        self.assertIsNotNone(res)
        self.assertEqual(_canon(res), _canon("C1CCCCC1"))

    def test_dioxolane_protection_preserves_carbonyl_map(self):
        # Cyclohexanone: the carbonyl carbon is IN A RING, the case the
        # remove_group + whole-ring-fragment idiom cannot express. Reduce C=O,
        # graft a tagged glycol tail onto the existing O, close the ring onto the
        # tagged oxygen -> 1,4-dioxaspiro[4.5]decane, carbonyl carbon retained.
        mapped = get_mapped_smiles("O=C1CCCCC1")  # O=map1, carbonyl C=map2
        ops = [
            {"op": "change_bond_order", "map_a": 2, "map_b": 1, "delta": -1},
            {"op": "add_group", "map_idx": 1, "fragment_smiles": "*CC[O:99]", "order": 1},
            {"op": "add_bond", "map_a": 2, "map_b": 99, "order": 1},
        ]
        res, err = execute_ops(ops, mapped)
        self.assertIsNone(err)
        self.assertEqual(_canon(res), _canon("C1CCC2(CC1)OCCO2"))
        # The carbonyl carbon (map 2) and original oxygen (map 1) survive.
        self.assertTrue({1, 2}.issubset(_map_nums(res)))

    def test_untagged_new_atom_reference_is_rejected(self):
        # Regression for the silent-corruption bug: acetone -> attempt a
        # dioxolane by adding an UNtagged glycol tail (its atoms all get map 0)
        # then referencing map 0 to close the ring. Previously this resolved to
        # an arbitrary map-0 atom and produced an epoxide with no error; now the
        # ambiguous reference is caught.
        mapped = get_mapped_smiles("CC(=O)C")  # carbonyl C=map2, O=map3
        ops = [
            {"op": "change_bond_order", "map_a": 2, "map_b": 3, "delta": -1},
            {"op": "add_group", "map_idx": 3, "fragment_smiles": "*CCO", "order": 1},
            {"op": "add_bond", "map_a": 2, "map_b": 0, "order": 1},
        ]
        res, err = execute_ops(ops, mapped)
        self.assertIsNone(res)
        self.assertIsNotNone(err)
        self.assertIn("mbiguous", err)  # "Ambiguous map number 0: ..."

    def test_large_target_over_99_atoms_tag_collision_and_safe_tag(self):
        # A 105-carbon chain maps atoms 1..105. A naive :99 tag DUPLICATES the
        # pre-existing atom 99. add_group allows the duplicate (route_ops plans
        # legitimately reuse product map numbers on re-attached groups); the danger
        # only materializes if a later op *references* the duplicated map, which
        # _resolve_map then rejects as ambiguous. A tag above the largest existing
        # map number (150) is safe and can close a ring.
        chain = "C" * 105
        mapped = get_mapped_smiles(chain)
        self.assertEqual(max(_map_nums(mapped)), 105)

        # Colliding tag that IS referenced later -> caught as ambiguous.
        collide = [
            {"op": "add_group", "map_idx": 1, "fragment_smiles": "*CC[O:99]", "order": 1},
            {"op": "add_bond", "map_a": 3, "map_b": 99, "order": 1},
        ]
        res, err = execute_ops(collide, mapped)
        self.assertIsNone(res)
        self.assertIsNotNone(err)
        self.assertIn("mbiguous", err)  # _resolve_map: ambiguous map number 99

        # A colliding tag that is never referenced is harmless (no error) — this is
        # what historical route_ops plans rely on.
        harmless = [
            {"op": "add_group", "map_idx": 1, "fragment_smiles": "*CC[O:99]", "order": 1},
        ]
        res_h, err_h = execute_ops(harmless, mapped)
        self.assertIsNone(err_h)
        self.assertIsNotNone(res_h)

        # Safe tag above the existing maximum -> ring closes.
        safe = [
            {"op": "add_group", "map_idx": 1, "fragment_smiles": "*CC[O:150]", "order": 1},
            {"op": "add_bond", "map_a": 3, "map_b": 150, "order": 1},
        ]
        res2, err2 = execute_ops(safe, mapped)
        self.assertIsNone(err2)
        self.assertIsNotNone(res2)
        result_mol = Chem.MolFromSmiles(res2)
        # Explicitly initialize ring perception; RDKit 2023.09 can leave the
        # RingInfo cache empty for this very long mapped SMILES after parsing.
        self.assertGreater(len(Chem.GetSymmSSSR(result_mol)), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
