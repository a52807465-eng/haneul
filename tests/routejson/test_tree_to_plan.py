"""Tests for tree_to_op_plan — lean plan structure only.

Detailed frontier / route rendering lives in ``test_render_plan.py``.
"""

from __future__ import annotations

import unittest

from rdkit import RDLogger

from synthex.routejson import tree_to_op_plan

RDLogger.DisableLog("rdApp.*")


def _mol(smiles: str, in_stock: bool = False, children=None) -> dict:
    return {
        "type": "mol",
        "smiles": smiles,
        "is_chemical": True,
        "in_stock": in_stock,
        "children": children or [],
    }


def _rxn(
    reaction_smiles: str,
    ops: list[dict],
    mapped_reaction_smiles: str,
    rationale: str = "",
    children=None,
    from_cache: bool = False,
    extra_meta: dict | None = None,
) -> dict:
    meta_key = "original_strategy" if from_cache else "reaction_prompt"
    metadata = {
        "operations": ops,
        "mapped_reaction_smiles": mapped_reaction_smiles,
        meta_key: rationale,
        "from_cache": from_cache,
    }
    if extra_meta:
        metadata.update(extra_meta)
    return {
        "type": "reaction",
        "smiles": reaction_smiles,
        "is_reaction": True,
        "metadata": metadata,
        "children": children or [],
    }


class TestTreeToOpPlan(unittest.TestCase):
    def test_single_step_lean_output(self):
        target_mapped = "[OH:1][CH2:2][CH2:3][OH:4]"
        reactant_mapped = (
            "[O:1]([CH2:2][CH2:3][OH:4])[Si:5]([CH3:6])([CH3:7])[C:8]([CH3:9])([CH3:10])[CH3:11]"
        )
        tree = _mol(
            "OCCO",
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
        plan = tree_to_op_plan(tree)
        # Top-level keys are minimal: just target + steps.
        self.assertEqual(set(plan.keys()), {"target", "steps"})
        self.assertEqual(plan["target"], target_mapped)
        self.assertEqual(len(plan["steps"]), 1)
        s1 = plan["steps"][0]
        # Step has id, desc, ops only (no rendered).
        self.assertEqual(s1["id"], "S1")
        self.assertEqual(s1["desc"], "protect terminal OH as TBS ether")
        self.assertEqual(s1["ops"][0]["op"], "add_group")
        self.assertNotIn("rendered", s1)

    def test_step_rationale_and_conditions_propagate_when_present(self):
        target_mapped = "[OH:1][CH3:2]"
        reactant_mapped = "[OH:1][CH3:2]"  # toy
        tree = _mol(
            "CO",
            children=[
                _rxn(
                    reaction_smiles="CO>>CO",
                    ops=[{"op": "break_bond", "map_a": 1, "map_b": 2}],
                    mapped_reaction_smiles=f"{target_mapped}>>{reactant_mapped}",
                    rationale="short prompt",
                    extra_meta={
                        "step_rationale": "longer strategic reasoning here",
                        "reaction_conditions": "anhydrous THF, 0C",
                    },
                    children=[_mol("C"), _mol("O")],
                ),
            ],
        )
        plan = tree_to_op_plan(tree)
        s = plan["steps"][0]
        self.assertEqual(s["desc"], "short prompt")
        self.assertEqual(s["step_rationale"], "longer strategic reasoning here")
        self.assertEqual(s["reaction_conditions"], "anhydrous THF, 0C")

    def test_optional_fields_omitted_when_absent(self):
        target_mapped = "[OH:1][CH3:2]"
        reactant_mapped = "[OH:1][CH3:2]"
        tree = _mol(
            "CO",
            children=[
                _rxn(
                    reaction_smiles="CO>>CO",
                    ops=[{"op": "break_bond", "map_a": 1, "map_b": 2}],
                    mapped_reaction_smiles=f"{target_mapped}>>{reactant_mapped}",
                    rationale="short prompt",
                    children=[_mol("C"), _mol("O")],
                ),
            ],
        )
        s = tree_to_op_plan(tree)["steps"][0]
        self.assertNotIn("step_rationale", s)
        self.assertNotIn("reaction_conditions", s)

    def test_missing_ops_falls_back_to_placeholder(self):
        target_mapped = "[CH3:1][OH:2]"
        reactant_mapped = "[CH3:1][O:2][CH3:3]"
        tree = _mol(
            "CO",
            children=[
                {
                    "type": "reaction",
                    "smiles": "CO>>COC",
                    "is_reaction": True,
                    "metadata": {
                        "mapped_reaction_smiles": f"{target_mapped}>>{reactant_mapped}",
                        "original_strategy": "methylation (no ops — legacy cache)",
                        "from_cache": True,
                    },
                    "children": [_mol("COC", in_stock=True)],
                }
            ],
        )
        plan = tree_to_op_plan(tree)
        self.assertEqual(plan["steps"][0]["ops"][0]["op"], "__interpolated__")

    def test_steps_in_dfs_preorder(self):
        # Two-step chain: S1 expands root, S2 expands S1's child.
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

        tree = _mol(
            "OCCO",
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
        plan = tree_to_op_plan(tree)
        self.assertEqual([s["id"] for s in plan["steps"]], ["S1", "S2"])
        self.assertEqual(plan["steps"][0]["desc"], "protect OH #1")
        self.assertEqual(plan["steps"][1]["desc"], "protect OH #2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
