"""Tests for op_plan_to_tree + plan↔tree round-trip."""

from __future__ import annotations

import unittest

from rdkit import Chem, RDLogger

from synthex.routejson import op_plan_to_tree, tree_to_op_plan

RDLogger.DisableLog("rdApp.*")


def _canon(smi: str) -> str:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else smi


def _collect_mol_smis(node, out=None):
    """Walk a route-tree dict, collecting canonical SMILES of every mol node
    in DFS pre-order."""
    if out is None:
        out = []
    if node.get("type") == "mol":
        out.append(_canon(node.get("smiles", "")))
    for c in node.get("children") or []:
        _collect_mol_smis(c, out)
    return out


def _mol(smiles: str, in_stock: bool = False, children=None) -> dict:
    return {
        "type": "mol",
        "smiles": smiles,
        "is_chemical": True,
        "in_stock": in_stock,
        "children": children or [],
    }


def _rxn(
    reaction_smiles, ops, mapped_reaction_smiles, rationale="", children=None, from_cache=False
):
    meta_key = "original_strategy" if from_cache else "reaction_prompt"
    return {
        "type": "reaction",
        "smiles": reaction_smiles,
        "is_reaction": True,
        "metadata": {
            "operations": ops,
            "mapped_reaction_smiles": mapped_reaction_smiles,
            meta_key: rationale,
            "from_cache": from_cache,
        },
        "children": children or [],
    }


class TestOpPlanToTree(unittest.TestCase):
    def test_single_step_protection(self):
        # TBS-protect map 1 of ethylene glycol.
        plan = {
            "target": "[OH:1][CH2:2][CH2:3][OH:4]",
            "config": {"band_base": 100, "band_size": 100},
            "steps": [
                {
                    "id": "S1",
                    "desc": "protect OH #1",
                    "ops": [
                        {"op": "add_group", "map_idx": 1, "fragment_smiles": "*[Si](C)(C)C(C)(C)C"}
                    ],
                },
            ],
        }
        tree = op_plan_to_tree(plan)
        self.assertEqual(tree["type"], "mol")
        # Root is the target (unmapped canonical).
        self.assertEqual(tree["smiles"], _canon("OCCO"))
        # One reaction child.
        self.assertEqual(len(tree["children"]), 1)
        rxn = tree["children"][0]
        self.assertEqual(rxn["type"], "reaction")
        self.assertEqual(rxn["metadata"]["operations"], plan["steps"][0]["ops"])
        self.assertIn(">>", rxn["metadata"]["mapped_reaction_smiles"])
        # One reactant mol child (single TBS-protected product, no fragmentation).
        self.assertEqual(len(rxn["children"]), 1)
        product = rxn["children"][0]
        self.assertEqual(product["type"], "mol")
        self.assertEqual(
            product["smiles"],
            _canon("OCCO[Si](C)(C)C(C)(C)C"),
        )

    def test_break_bond_splits_frontier(self):
        # Break ethane → two methanes.
        plan = {
            "target": "[CH3:1][CH3:2]",
            "config": {"band_base": 100, "band_size": 100},
            "steps": [
                {
                    "id": "S1",
                    "desc": "toy bond cleavage",
                    "ops": [{"op": "break_bond", "map_a": 1, "map_b": 2}],
                },
            ],
        }
        tree = op_plan_to_tree(plan)
        rxn = tree["children"][0]
        self.assertEqual(len(rxn["children"]), 2)
        for child in rxn["children"]:
            self.assertEqual(child["smiles"], _canon("C"))

    def test_two_step_chain(self):
        # S1: protect map 1; S2: protect map 4 (of the resulting mol).
        plan = {
            "target": "[OH:1][CH2:2][CH2:3][OH:4]",
            "config": {"band_base": 100, "band_size": 100},
            "steps": [
                {
                    "id": "S1",
                    "desc": "protect OH #1",
                    "ops": [
                        {"op": "add_group", "map_idx": 1, "fragment_smiles": "*[Si](C)(C)C(C)(C)C"}
                    ],
                },
                {
                    "id": "S2",
                    "desc": "protect OH #4",
                    "ops": [
                        {"op": "add_group", "map_idx": 4, "fragment_smiles": "*[Si](C)(C)C(C)(C)C"}
                    ],
                },
            ],
        }
        tree = op_plan_to_tree(plan)
        # Walk to depth 2: root → rxn1 → mol1 → rxn2 → mol2.
        mol1 = tree["children"][0]["children"][0]
        self.assertEqual(len(mol1["children"]), 1)
        rxn2 = mol1["children"][0]
        self.assertEqual(rxn2["metadata"]["operations"][0]["map_idx"], 4)
        mol2 = rxn2["children"][0]
        # Final mol: TBS on both ends.
        self.assertEqual(
            mol2["smiles"],
            _canon("CC(C)(C)[Si](C)(C)OCCO[Si](C)(C)C(C)(C)C"),
        )

    def test_stock_check_callback_used(self):
        plan = {
            "target": "[OH:1][CH2:2][CH2:3][OH:4]",
            "config": {"band_base": 100, "band_size": 100},
            "steps": [
                {
                    "id": "S1",
                    "desc": "trivial protect",
                    "ops": [
                        {"op": "add_group", "map_idx": 1, "fragment_smiles": "*[Si](C)(C)C(C)(C)C"}
                    ],
                },
            ],
        }
        # Mark everything in stock.
        tree = op_plan_to_tree(plan, stock_check=lambda _smi: True)
        # Root + reactant should both be flagged in_stock.
        self.assertTrue(tree["in_stock"])
        self.assertTrue(tree["children"][0]["children"][0]["in_stock"])

    def test_interpolated_placeholder_rejected(self):
        plan = {
            "target": "[OH:1][CH3:2]",
            "config": {"band_base": 100, "band_size": 100},
            "steps": [
                {"id": "S1", "desc": "", "ops": [{"op": "__interpolated__", "reason": "legacy"}]}
            ],
        }
        with self.assertRaises(ValueError) as cm:
            op_plan_to_tree(plan)
        self.assertIn("__interpolated__", str(cm.exception))

    def test_unresolvable_op_maps_errors(self):
        # Op references atom map 99 which doesn't exist in the target.
        plan = {
            "target": "[OH:1][CH3:2]",
            "config": {"band_base": 100, "band_size": 100},
            "steps": [
                {"id": "S1", "desc": "", "ops": [{"op": "break_bond", "map_a": 1, "map_b": 99}]}
            ],
        }
        with self.assertRaises(ValueError) as cm:
            op_plan_to_tree(plan)
        self.assertIn("S1", str(cm.exception))


class TestRoundTrip(unittest.TestCase):
    """tree_to_op_plan ∘ op_plan_to_tree on synthetic trees."""

    def test_single_step_roundtrip(self):
        target_mapped = "[OH:1][CH2:2][CH2:3][OH:4]"
        reactant_mapped = (
            "[O:1]([CH2:2][CH2:3][OH:4])[Si:5]([CH3:6])([CH3:7])[C:8]([CH3:9])([CH3:10])[CH3:11]"
        )
        original = _mol(
            "OCCO",
            in_stock=False,
            children=[
                _rxn(
                    reaction_smiles="OCCO>>CC(C)(C)[Si](C)(C)OCCO",
                    ops=[
                        {"op": "add_group", "map_idx": 1, "fragment_smiles": "*[Si](C)(C)C(C)(C)C"}
                    ],
                    mapped_reaction_smiles=f"{target_mapped}>>{reactant_mapped}",
                    rationale="protect terminal OH as TBS ether",
                    children=[_mol("CC(C)(C)[Si](C)(C)OCCO", in_stock=True)],
                ),
            ],
        )
        plan = tree_to_op_plan(original)
        reconstructed = op_plan_to_tree(plan)
        # Compare as multisets: RDKit canonicalisation may re-order
        # multi-fragment reactant lists, but the set of mols visited
        # across the route must match.
        self.assertEqual(
            sorted(_collect_mol_smis(reconstructed)), sorted(_collect_mol_smis(original))
        )

    def test_break_bond_roundtrip(self):
        original = _mol(
            "CC",
            in_stock=False,
            children=[
                _rxn(
                    reaction_smiles="CC>>C.C",
                    ops=[{"op": "break_bond", "map_a": 1, "map_b": 2}],
                    mapped_reaction_smiles="[CH3:1][CH3:2]>>[CH4:1].[CH4:2]",
                    rationale="bond cleavage (toy)",
                    children=[
                        _mol("C", in_stock=True),
                        _mol("C", in_stock=True),
                    ],
                ),
            ],
        )
        plan = tree_to_op_plan(original)
        reconstructed = op_plan_to_tree(plan)
        # Compare as multisets: RDKit canonicalisation may re-order
        # multi-fragment reactant lists, but the set of mols visited
        # across the route must match.
        self.assertEqual(
            sorted(_collect_mol_smis(reconstructed)), sorted(_collect_mol_smis(original))
        )

    def test_two_step_roundtrip(self):
        target_mapped = "[OH:1][CH2:2][CH2:3][OH:4]"
        s1_reactant_mapped = (
            "[O:1]([CH2:2][CH2:3][OH:4])[Si:5]([CH3:6])([CH3:7])[C:8]([CH3:9])([CH3:10])[CH3:11]"
        )
        s2_reactant_mapped = (
            "[O:1]([CH2:2][CH2:3][O:4][Si:12]([CH3:13])([CH3:14])"
            "[C:15]([CH3:16])([CH3:17])[CH3:18])[Si:5]([CH3:6])([CH3:7])"
            "[C:8]([CH3:9])([CH3:10])[CH3:11]"
        )
        s1_plain = "CC(C)(C)[Si](C)(C)OCCO"
        s2_plain = "CC(C)(C)[Si](C)(C)OCCO[Si](C)(C)C(C)(C)C"

        original = _mol(
            "OCCO",
            in_stock=False,
            children=[
                _rxn(
                    reaction_smiles=f"OCCO>>{s1_plain}",
                    ops=[
                        {"op": "add_group", "map_idx": 1, "fragment_smiles": "*[Si](C)(C)C(C)(C)C"}
                    ],
                    mapped_reaction_smiles=f"{target_mapped}>>{s1_reactant_mapped}",
                    rationale="protect OH #1",
                    children=[
                        _mol(
                            s1_plain,
                            in_stock=False,
                            children=[
                                _rxn(
                                    reaction_smiles=f"{s1_plain}>>{s2_plain}",
                                    ops=[
                                        {
                                            "op": "add_group",
                                            "map_idx": 4,
                                            "fragment_smiles": "*[Si](C)(C)C(C)(C)C",
                                        }
                                    ],
                                    mapped_reaction_smiles=f"{s1_reactant_mapped}>>{s2_reactant_mapped}",
                                    rationale="protect OH #2",
                                    children=[_mol(s2_plain, in_stock=True)],
                                ),
                            ],
                        )
                    ],
                ),
            ],
        )
        plan = tree_to_op_plan(original)
        reconstructed = op_plan_to_tree(plan)
        # Compare as multisets: RDKit canonicalisation may re-order
        # multi-fragment reactant lists, but the set of mols visited
        # across the route must match.
        self.assertEqual(
            sorted(_collect_mol_smis(reconstructed)), sorted(_collect_mol_smis(original))
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
