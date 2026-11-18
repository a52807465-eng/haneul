"""Tests for the fast tree-json loader.

The fast loader bypasses ``LLMGuidedBeamSearchTree.from_json`` (and the
whole Configuration with its template engine + 22M-compound stock load),
parsing the JSON directly into route_dicts. These tests pin both the
output shape and the speed win against the canonical loader.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from rdkit import RDLogger

from synthex.routejson import (
    load_route_trees_from_tree_json,
    load_solved_route_trees_from_tree_json,
    tree_to_op_plan,
)

RDLogger.DisableLog("rdApp.*")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_TREE = _REPO_ROOT / "data" / "example_npatlas" / "npa035920_s1" / "tree.0.llm_guided.json"


def _make_minimal_tree_json(tmp_path: Path) -> Path:
    """Hand-built minimal tree.json: one root mol → one expansion → two reactants.

    Mirrors the on-disk schema from MoleculeSerializer + serialize_action +
    LLMGuidedMctsNode.serialize.
    """
    # mol ids are arbitrary ints (id() in production).
    target_id = 1000
    reactant_a_id = 2000
    reactant_b_id = 3000

    target_mapped = "[CH3:1][CH3:2]"
    react_a_mapped = "[CH4:1]"
    react_b_mapped = "[CH4:2]"

    molecules = {
        # Root mol: parent=None → 'smiles' is original (unmapped).
        str(target_id): {"smiles": "CC", "class": "TreeMolecule", "parent": None, "transform": 0},
        str(reactant_a_id): {
            "smiles": react_a_mapped,
            "class": "TreeMolecule",
            "parent": target_id,
            "transform": 1,
        },
        str(reactant_b_id): {
            "smiles": react_b_mapped,
            "class": "TreeMolecule",
            "parent": target_id,
            "transform": 1,
        },
    }
    action = {
        "mol": target_id,
        "index": 0,
        "reactants": [[reactant_a_id, reactant_b_id]],
        "class": "synthelite.chem.SmilesBasedRetroReaction",
        "metadata": {
            "policy_name": "test",
            "operations": [{"op": "break_bond", "map_a": 1, "map_b": 2}],
            "mapped_reaction_smiles": f"{target_mapped}>>{react_a_mapped}.{react_b_mapped}",
            "reaction_prompt": "toy bond cleavage",
        },
    }

    # The expanded child is a terminal leaf (no further children_actions).
    leaf_node = {
        "state": {"mols": [reactant_a_id, reactant_b_id]},
        "children_actions": [],
        "children": [],
        "children_values": [],
        "children_priors": [],
        "children_heuristic_scores": [],
        "children_visitations": [],
        "is_expanded": False,
        "is_expandable": False,
        "synthesis_dynamic_context": None,
    }
    root_node = {
        "state": {"mols": [target_id]},
        "children_actions": [action],
        "children": [leaf_node],
        "children_values": [],
        "children_priors": [],
        "children_heuristic_scores": [],
        "children_visitations": [],
        "is_expanded": True,
        "is_expandable": False,
        "synthesis_dynamic_context": None,
    }
    tree_json = {"tree": root_node, "molecules": molecules}
    path = tmp_path / "tree.json"
    path.write_text(json.dumps(tree_json))
    return path


class TestLoaderShape(unittest.TestCase):
    def test_minimal_tree_yields_one_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_minimal_tree_json(Path(tmp))
            routes = load_route_trees_from_tree_json(path)
        self.assertEqual(len(routes), 1)
        r = routes[0]
        self.assertEqual(r["type"], "mol")
        self.assertEqual(r["smiles"], "CC")
        rxn = r["children"][0]
        self.assertEqual(rxn["type"], "reaction")
        self.assertEqual(rxn["metadata"]["operations"][0]["op"], "break_bond")
        # Two reactant children, both methanes.
        self.assertEqual(len(rxn["children"]), 2)
        for child in rxn["children"]:
            self.assertEqual(child["smiles"], "C")

    def test_stock_check_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_minimal_tree_json(Path(tmp))
            # Mark methane in stock.
            routes = load_route_trees_from_tree_json(
                path,
                stock_check=lambda s: s == "C",
            )
        rxn = routes[0]["children"][0]
        for child in rxn["children"]:
            self.assertTrue(child["in_stock"], f"{child['smiles']} should be in stock")
        # Root (ethane) not in stock.
        self.assertFalse(routes[0]["in_stock"])

    def test_solved_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_minimal_tree_json(Path(tmp))
            # Methane in stock → route is solved.
            solved = load_solved_route_trees_from_tree_json(
                path,
                stock_check=lambda s: s == "C",
            )
            self.assertEqual(len(solved), 1)
            # Methane NOT in stock → no solved routes.
            none = load_solved_route_trees_from_tree_json(
                path,
                stock_check=lambda s: False,
            )
            self.assertEqual(len(none), 0)

    def test_no_internal_keys_in_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_minimal_tree_json(Path(tmp))
            routes = load_route_trees_from_tree_json(path)

        def assert_clean(node):
            for k in ("_mapped_smi", "_mol_id"):
                self.assertNotIn(k, node, f"internal key {k} leaked in output")
            for c in node.get("children") or []:
                assert_clean(c)

        assert_clean(routes[0])


@unittest.skipUnless(_REAL_TREE.exists(), f"real tree json not present at {_REAL_TREE}")
class TestLoaderRealData(unittest.TestCase):
    """Speed + correctness against the real npa035920_s1 tree.json."""

    def test_loads_and_yields_routes(self):
        t0 = time.time()
        routes = load_route_trees_from_tree_json(_REAL_TREE)
        elapsed = time.time() - t0
        # No assertion on count (depends on the search), but at least one
        # leaf should be present.
        self.assertGreater(len(routes), 0)
        # Speed budget: pure-JSON loader must be well under one second on a
        # tree of this size (the canonical loader pulls in 22M compounds
        # and takes ~minute). Generous bound to keep CI stable.
        self.assertLess(
            elapsed,
            5.0,
            f"fast loader took {elapsed:.3f}s — config-free path should be much faster",
        )

    def test_routes_convert_to_op_plan(self):
        routes = load_route_trees_from_tree_json(_REAL_TREE)
        plan = tree_to_op_plan(routes[0])
        self.assertIn("target", plan)
        self.assertIn("steps", plan)
        self.assertGreater(len(plan["steps"]), 0)
        # Every step from a fresh code-ops run should carry ops.
        for s in plan["steps"]:
            self.assertGreater(len(s["ops"]), 0, f"step {s['id']} missing ops")


if __name__ == "__main__":
    unittest.main(verbosity=2)
