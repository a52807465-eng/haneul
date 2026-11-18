"""AiZynthFinder integration tests for end-to-end atom-map propagation.

Replicates the data flow in LLMCodeOpsToReactionStrategy._expand_molecule
without invoking the LLM: take a TreeMolecule, run execute_ops on its
mapped_smiles, wrap the result in a SmilesBasedRetroReaction, and check
that the resulting mapped_reaction_smiles has LHS=RHS atom-map consistency.

This is the contract route_ops depends on: every step in a tree shares one
global atom-map space across product (LHS) and reactants (RHS).
"""

from __future__ import annotations

import re
import unittest

from aizynthfinder.chem import SmilesBasedRetroReaction, TreeMolecule
from rdkit import RDLogger

from synthex.reactionjson.executor import execute_ops

RDLogger.DisableLog("rdApp.*")


_MAP_RE = re.compile(r":(\d+)\]")


def _maps_in(smi: str) -> set[int]:
    return {int(m) for m in _MAP_RE.findall(smi)}


def _step(mol: TreeMolecule, ops: list[dict]) -> tuple[SmilesBasedRetroReaction, TreeMolecule]:
    """Mirror LLMCodeOpsToReactionStrategy._expand_molecule: execute ops on
    mol.mapped_smiles, build a SmilesBasedRetroReaction, force application,
    return (reaction, first reactant)."""
    reactants_smi, err = execute_ops(ops, mol.mapped_smiles)
    assert err is None, f"execute_ops error: {err}"
    rxn = SmilesBasedRetroReaction(
        mol=mol, metadata={"policy_name": "test"}, reactants_str=reactants_smi
    )
    _ = rxn.reactants
    # Single-reactant case for these tests; callers using multi-reactant ops
    # (break_bond producing fragments) read rxn.reactants[0] themselves.
    first = rxn.reactants[0][0]
    return rxn, first


class TestChainMapPropagation(unittest.TestCase):
    def test_depth1_tbs_protection_preserves_maps(self):
        m0 = TreeMolecule(parent=None, smiles="OCCO", sanitize=True)
        # m0 should have maps 1..4.
        self.assertEqual(_maps_in(m0.mapped_smiles), {1, 2, 3, 4})

        ops = [{"op": "add_group", "map_idx": 1, "fragment_smiles": "*[Si](C)(C)C(C)(C)C"}]
        rxn, m1 = _step(m0, ops)

        rxn_smi = rxn.mapped_reaction_smiles()
        lhs_smi, rhs_smi = rxn_smi.split(">>")
        lhs_maps = _maps_in(lhs_smi)
        rhs_maps = _maps_in(rhs_smi)

        # LHS map space must be a subset of RHS (every product atom carried
        # forward to the reactants); RHS may have additional new-atom maps.
        self.assertTrue(
            lhs_maps.issubset(rhs_maps),
            f"LHS maps {lhs_maps} not subset of RHS maps {rhs_maps}",
        )
        # No collisions with new atoms: new maps must be > max(lhs).
        new_maps = rhs_maps - lhs_maps
        if new_maps:
            self.assertGreater(min(new_maps), max(lhs_maps))

    def test_depth2_chain_consistent_global_maps(self):
        m0 = TreeMolecule(parent=None, smiles="OCCO", sanitize=True)
        original_maps = _maps_in(m0.mapped_smiles)

        # Step 1: protect the first OH.
        ops1 = [{"op": "add_group", "map_idx": 1, "fragment_smiles": "*[Si](C)(C)C(C)(C)C"}]
        _, m1 = _step(m0, ops1)
        m1_maps = _maps_in(m1.mapped_smiles)
        self.assertTrue(original_maps.issubset(m1_maps))

        # Step 2: protect the other OH.
        ops2 = [{"op": "add_group", "map_idx": 4, "fragment_smiles": "*[Si](C)(C)C(C)(C)C"}]
        _, m2 = _step(m1, ops2)
        m2_maps = _maps_in(m2.mapped_smiles)

        # Step-2's product maps must be a superset of step-1's reactant maps.
        self.assertTrue(m1_maps.issubset(m2_maps))

    def test_map_1_traces_same_atom_through_chain(self):
        # Identity check: map id 1 must point at the same atom across depths
        # (modulo bond changes the op applies to it).
        m0 = TreeMolecule(parent=None, smiles="OCCO", sanitize=True)
        # At depth 0: map 1 is one of the OH oxygens (one H).
        a0 = next(a for a in m0.mapped_mol.GetAtoms() if a.GetAtomMapNum() == 1)
        self.assertEqual(a0.GetSymbol(), "O")
        h_before = a0.GetNumExplicitHs() + a0.GetNumImplicitHs()
        self.assertEqual(h_before, 1)

        # Apply TBS to map 1; at depth 1 the same atom is now bonded to Si.
        ops = [{"op": "add_group", "map_idx": 1, "fragment_smiles": "*[Si](C)(C)C(C)(C)C"}]
        _, m1 = _step(m0, ops)
        a1 = next(a for a in m1.mapped_mol.GetAtoms() if a.GetAtomMapNum() == 1)
        self.assertEqual(a1.GetSymbol(), "O")
        neighbor_syms = {n.GetSymbol() for n in a1.GetNeighbors()}
        self.assertIn("Si", neighbor_syms)
        h_after = a1.GetNumExplicitHs() + a1.GetNumImplicitHs()
        self.assertEqual(h_after, 0, "OH proton should be gone after silylation")

    def test_break_bond_keeps_both_halves_mapped(self):
        # break_bond splits the molecule; both fragments must retain their
        # original maps within the multi-fragment reactants_str.
        m0 = TreeMolecule(parent=None, smiles="CCCC", sanitize=True)
        self.assertEqual(_maps_in(m0.mapped_smiles), {1, 2, 3, 4})

        ops = [{"op": "break_bond", "map_a": 2, "map_b": 3}]
        reactants_smi, err = execute_ops(ops, m0.mapped_smiles)
        self.assertIsNone(err)
        # Two fragments separated by '.', joint map set still 1..4.
        self.assertEqual(_maps_in(reactants_smi), {1, 2, 3, 4})

        rxn = SmilesBasedRetroReaction(
            mol=m0, metadata={"policy_name": "test"}, reactants_str=reactants_smi
        )
        _ = rxn.reactants
        # All produced reactant molecules together cover {1..4} with no
        # duplicates (each atom belongs to exactly one fragment).
        seen = set()
        for r in rxn.reactants[0]:
            r_maps = _maps_in(r.mapped_smiles)
            self.assertFalse(r_maps & seen, f"duplicate maps across fragments: {r_maps & seen}")
            seen |= r_maps
        self.assertEqual(seen, {1, 2, 3, 4})
