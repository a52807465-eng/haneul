"""Tests for render_plan — populates rendered.frontier + bottom route[]."""

from __future__ import annotations

import unittest

from rdkit import RDLogger

from synthex.routejson import render_plan

RDLogger.DisableLog("rdApp.*")


_PROTECT_OCCO = {
    "target": "[OH:1][CH2:2][CH2:3][OH:4]",
    "steps": [
        {
            "id": "S1",
            "desc": "protect OH #1",
            "ops": [{"op": "add_group", "map_idx": 1, "fragment_smiles": "*[Si](C)(C)C(C)(C)C"}],
        },
        {
            "id": "S2",
            "desc": "protect OH #4",
            "ops": [{"op": "add_group", "map_idx": 4, "fragment_smiles": "*[Si](C)(C)C(C)(C)C"}],
        },
    ],
}


class TestRenderPlan(unittest.TestCase):
    def test_render_all_populates_every_step(self):
        out = render_plan(_PROTECT_OCCO)
        self.assertEqual(len(out["steps"]), 2)
        for s in out["steps"]:
            self.assertIn("rendered", s)
            self.assertGreater(len(s["rendered"]["frontier"]), 0)

    def test_route_summary_present_and_unmapped(self):
        out = render_plan(_PROTECT_OCCO)
        self.assertEqual(len(out["route"]), 2)
        self.assertEqual(out["route"][0]["id"], "S1")
        smi = out["route"][0]["smiles"]
        self.assertIn(">>", smi)
        self.assertNotIn(":", smi)

    def test_input_not_mutated(self):
        before_keys = set(_PROTECT_OCCO.keys())
        _ = render_plan(_PROTECT_OCCO)
        self.assertEqual(set(_PROTECT_OCCO.keys()), before_keys)
        for s in _PROTECT_OCCO["steps"]:
            self.assertNotIn("rendered", s)

    def test_break_bond_grows_frontier(self):
        plan = {
            "target": "[CH3:1][CH3:2]",
            "steps": [
                {"id": "S1", "desc": "toy", "ops": [{"op": "break_bond", "map_a": 1, "map_b": 2}]},
            ],
        }
        out = render_plan(plan, stock_check=lambda s: s == "C")
        frontier = out["steps"][0]["rendered"]["frontier"]
        self.assertEqual(len(frontier), 2)
        self.assertTrue(all(f["in_stock"] for f in frontier))
        self.assertTrue(out["steps"][0]["rendered"]["solved"])

    def test_strip_frontier_maps(self):
        out = render_plan(_PROTECT_OCCO, strip_frontier_maps=True)
        smi = out["steps"][0]["rendered"]["frontier"][0]["smi"]
        self.assertNotIn(":", smi)

    def test_up_to_int_renders_only_first_n(self):
        out = render_plan(_PROTECT_OCCO, up_to=1)
        self.assertIn("rendered", out["steps"][0])
        self.assertNotIn("rendered", out["steps"][1])
        # Route truncated to match.
        self.assertEqual(len(out["route"]), 1)
        self.assertEqual(out["route"][0]["id"], "S1")

    def test_up_to_str_matches_step_id(self):
        plan = dict(_PROTECT_OCCO)
        # Three steps so we can target the middle.
        plan = {
            "target": "[OH:1][CH2:2][CH2:3][OH:4]",
            "steps": [
                {
                    "id": "S1",
                    "desc": "p1",
                    "ops": [
                        {"op": "add_group", "map_idx": 1, "fragment_smiles": "*[Si](C)(C)C(C)(C)C"}
                    ],
                },
                {
                    "id": "S2",
                    "desc": "p2",
                    "ops": [
                        {"op": "add_group", "map_idx": 4, "fragment_smiles": "*[Si](C)(C)C(C)(C)C"}
                    ],
                },
                {"id": "S3", "desc": "noop", "ops": []},  # not reached
            ],
        }
        out = render_plan(plan, up_to="S2")
        self.assertIn("rendered", out["steps"][0])
        self.assertIn("rendered", out["steps"][1])
        self.assertNotIn("rendered", out["steps"][2])
        self.assertEqual([r["id"] for r in out["route"]], ["S1", "S2"])

    def test_up_to_default_renders_all(self):
        out = render_plan(_PROTECT_OCCO, up_to=None)
        for s in out["steps"]:
            self.assertIn("rendered", s)

    def test_up_to_out_of_range_errors(self):
        with self.assertRaises(ValueError):
            render_plan(_PROTECT_OCCO, up_to=99)
        with self.assertRaises(ValueError):
            render_plan(_PROTECT_OCCO, up_to=0)
        with self.assertRaises(ValueError):
            render_plan(_PROTECT_OCCO, up_to="S99")

    def test_is_leaf_flag_present(self):
        out = render_plan(_PROTECT_OCCO)
        for s in out["steps"]:
            for entry in s["rendered"]["frontier"]:
                self.assertIn("is_leaf", entry)
                self.assertIsInstance(entry["is_leaf"], bool)

    def test_is_leaf_marks_only_terminal_mols(self):
        # Two-step protection: after S1 the intermediate appears in
        # frontier but is consumed by S2 → is_leaf=False at S1.
        # After S2 the final di-protected mol is in frontier → is_leaf=True.
        out = render_plan(_PROTECT_OCCO)
        s1_frontier = out["steps"][0]["rendered"]["frontier"]
        self.assertEqual(len(s1_frontier), 1)
        # The mol introduced at S1 gets disconnected by S2, so not a leaf.
        self.assertFalse(s1_frontier[0]["is_leaf"])

        s2_frontier = out["steps"][1]["rendered"]["frontier"]
        self.assertEqual(len(s2_frontier), 1)
        self.assertTrue(s2_frontier[0]["is_leaf"])

    def test_is_leaf_partial_render_uses_active_frontier(self):
        # With up_to=1, the rendered set's "final frontier" is whatever
        # S1 produces — every entry there counts as a leaf for the
        # rendered window.
        out = render_plan(_PROTECT_OCCO, up_to=1)
        s1_frontier = out["steps"][0]["rendered"]["frontier"]
        self.assertTrue(all(e["is_leaf"] for e in s1_frontier))

    def test_is_leaf_solved_vs_unsolved_distinction(self):
        # Build a plan whose final frontier contains both an in-stock and
        # an out-of-stock mol. Confirm is_leaf=True for both, but only
        # one carries in_stock=True.
        plan = {
            "target": "[CH3:1][CH3:2]",
            "steps": [
                {
                    "id": "S1",
                    "desc": "split",
                    "ops": [{"op": "break_bond", "map_a": 1, "map_b": 2}],
                },
            ],
        }
        out = render_plan(plan, stock_check=lambda s: s == "C")
        frontier = out["steps"][0]["rendered"]["frontier"]
        self.assertEqual(len(frontier), 2)
        for e in frontier:
            self.assertTrue(e["is_leaf"])
        # Both leaves should be solved since both are methane.
        self.assertTrue(all(e["in_stock"] for e in frontier))

    def test_interpolated_placeholder_rejected(self):
        plan = {
            "target": "[OH:1][CH3:2]",
            "steps": [
                {"id": "S1", "desc": "x", "ops": [{"op": "__interpolated__", "reason": "legacy"}]}
            ],
        }
        with self.assertRaises(ValueError) as cm:
            render_plan(plan)
        self.assertIn("__interpolated__", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
