"""Tests for execute_ops atom-map preservation.

The route_ops global-map design needs reactant SMILES to carry the same atom
maps as the product. execute_ops now preserves maps unconditionally; new
atoms introduced by ops (e.g. add_group) come out at map=0 and downstream
code (TreeMolecule._update_unmapped_atom_num, or the route_ops executor)
assigns offset map ids.
"""

from __future__ import annotations

import re
import unittest

from rdkit import RDLogger

from synthex.reactionjson.executor import execute_ops, get_mapped_smiles

RDLogger.DisableLog("rdApp.*")


_MAP_RE = re.compile(r":(\d+)\]")


def _maps_in(smi: str) -> set[int]:
    return {int(m) for m in _MAP_RE.findall(smi)}


class TestExecuteOpsPreserveMaps(unittest.TestCase):
    def test_preserves_input_maps(self):
        mapped = get_mapped_smiles("CCO")  # [CH3:1][CH2:2][OH:3]
        ops = [{"op": "add_group", "map_idx": 3, "fragment_smiles": "*[C:100]"}]
        result, err = execute_ops(ops, mapped)
        self.assertIsNone(err)
        out_maps = _maps_in(result)
        # Original maps (1, 2, 3) all survive; frag's declared map (100) too.
        for expected in (1, 2, 3, 100):
            self.assertIn(expected, out_maps, f"map {expected} missing from {result!r}")

    def test_new_atoms_with_bare_frag_get_map_zero(self):
        # Bare add_group frag (no maps in frag SMILES) → new atoms map=0.
        mapped = get_mapped_smiles("CCO")
        ops = [{"op": "add_group", "map_idx": 3, "fragment_smiles": "*C"}]
        result, err = execute_ops(ops, mapped)
        self.assertIsNone(err)
        out_maps = _maps_in(result)
        # Original maps survive.
        self.assertEqual(out_maps, {1, 2, 3})

    def test_break_bond_keeps_all_maps(self):
        mapped = get_mapped_smiles("CCCC")  # [CH3:1][CH2:2][CH2:3][CH3:4]
        ops = [{"op": "break_bond", "map_a": 2, "map_b": 3}]
        result, err = execute_ops(ops, mapped)
        self.assertIsNone(err)
        self.assertEqual(_maps_in(result), {1, 2, 3, 4})

    def test_remove_group_drops_removed_atom_maps(self):
        mapped = get_mapped_smiles("CCO")  # [CH3:1][CH2:2][OH:3]
        # Remove the methyl group rooted at map 1, keeping map 2.
        ops = [{"op": "remove_group", "map_idx": 2, "map_root_idx": 1}]
        result, err = execute_ops(ops, mapped)
        self.assertIsNone(err, f"unexpected error: {err}")
        out_maps = _maps_in(result)
        self.assertNotIn(1, out_maps)
        self.assertIn(2, out_maps)
        self.assertIn(3, out_maps)

    def test_chains_across_multiple_ops(self):
        mapped = get_mapped_smiles("CCO")
        ops = [
            {"op": "add_group", "map_idx": 3, "fragment_smiles": "*[C:50]"},
            {"op": "break_bond", "map_a": 1, "map_b": 2},
        ]
        result, err = execute_ops(ops, mapped)
        self.assertIsNone(err, f"unexpected error: {err}")
        for expected in (1, 2, 3, 50):
            self.assertIn(expected, _maps_in(result))


if __name__ == "__main__":
    unittest.main(verbosity=2)
