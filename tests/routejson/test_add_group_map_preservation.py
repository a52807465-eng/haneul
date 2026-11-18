"""Tests for add_group atom-map preservation.

Route_ops needs add_group to round-trip atom maps so the executor can chain
ops across a route while keeping a single global map space. The previous
implementation overwrote the fragment anchor's map with a sentinel and reset
it to 0; the patch saves and restores the original map.
"""

from __future__ import annotations

import unittest

from rdkit import Chem, RDLogger

from synthex.reactionjson.operations import add_group

RDLogger.DisableLog("rdApp.*")


def _smi(s: str) -> Chem.Mol:
    m = Chem.MolFromSmiles(s)
    assert m is not None, f"bad SMILES: {s!r}"
    return m


class TestAddGroupMapPreservation(unittest.TestCase):
    def test_no_maps_in_means_no_maps_out(self):
        # Bare frag + unmapped main mol → all output atoms have map 0.
        m = _smi("CO")
        out = add_group(m, 1, "*CC")
        for atom in out.GetAtoms():
            self.assertEqual(atom.GetAtomMapNum(), 0)

    def test_main_mol_maps_survive(self):
        m = _smi("CO")
        m.GetAtomWithIdx(0).SetAtomMapNum(7)
        m.GetAtomWithIdx(1).SetAtomMapNum(8)
        out = add_group(m, 1, "*CC")
        maps = {a.GetAtomMapNum() for a in out.GetAtoms()}
        self.assertIn(7, maps)
        self.assertIn(8, maps)

    def test_frag_anchor_map_preserved(self):
        # The patched code saves and restores the anchor's original map
        # number. Previously this round-tripped through the sentinel and
        # came out as 0.
        m = _smi("CO")
        out = add_group(m, 1, "*[C:200](C)C")
        maps = {a.GetAtomMapNum() for a in out.GetAtoms()}
        self.assertIn(200, maps, f"anchor map 200 lost; output maps: {maps}")
        anchor = next(a for a in out.GetAtoms() if a.GetAtomMapNum() == 200)
        # Anchor must be bonded to the main-mol attachment atom (O of CO).
        self.assertIn("O", {n.GetSymbol() for n in anchor.GetNeighbors()})

    def test_frag_nonanchor_maps_propagate(self):
        m = _smi("CO")
        out = add_group(m, 1, "*[C:200]([C:201])[C:202]")
        maps = {a.GetAtomMapNum() for a in out.GetAtoms()}
        for expected in (200, 201, 202):
            self.assertIn(expected, maps)

    def test_main_and_frag_maps_coexist(self):
        m = _smi("CO")
        m.GetAtomWithIdx(0).SetAtomMapNum(1)
        m.GetAtomWithIdx(1).SetAtomMapNum(2)
        out = add_group(m, 1, "*[C:100]([C:101])[C:102]")
        maps = {a.GetAtomMapNum() for a in out.GetAtoms()}
        self.assertTrue({1, 2, 100, 101, 102}.issubset(maps))

    def test_frag_anchor_zero_map_stays_zero(self):
        # Bare anchor in frag → anchor in output still has map 0 (not
        # accidentally set to the sentinel 999).
        m = _smi("CO")
        out = add_group(m, 1, "*CC")
        # Find a new C atom (the one bonded to the O w/ no map originally).
        # All new atoms must have map 0.
        for atom in out.GetAtoms():
            self.assertNotEqual(atom.GetAtomMapNum(), 999)


if __name__ == "__main__":
    unittest.main(verbosity=2)
