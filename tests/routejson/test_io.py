"""Tests for save_plan / load_plan round-trip."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rdkit import RDLogger

from synthex.routejson import (
    load_plan,
    plan_from_yaml,
    plan_to_yaml,
    save_plan,
)

RDLogger.DisableLog("rdApp.*")


_SAMPLE_PLAN = {
    "target": "[OH:1][CH2:2][CH2:3][OH:4]",
    "config": {"band_base": 100, "band_size": 100},
    "steps": [
        {
            "id": "S1",
            "desc": "Protect first OH as TBS ether",
            "ops": [
                {"op": "add_group", "map_idx": 1, "fragment_smiles": "*[Si](C)(C)C(C)(C)C"},
            ],
            "rendered": {
                "frontier": [
                    {"smi": "[O:1][CH2:2][CH2:3][OH:4]...", "in_stock": False},
                ],
                "solved": False,
            },
        },
    ],
}


class TestPlanIO(unittest.TestCase):
    def test_yaml_roundtrip_preserves_structure(self):
        s = plan_to_yaml(_SAMPLE_PLAN)
        self.assertIn("target:", s)
        self.assertIn("S1", s)
        back = plan_from_yaml(s)
        self.assertEqual(back, _SAMPLE_PLAN)

    def test_save_yaml_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "plan.yaml"
            save_plan(_SAMPLE_PLAN, p)
            self.assertTrue(p.exists())
            content = p.read_text()
            self.assertIn("target:", content)
            self.assertNotIn('"target":', content)  # not JSON
            back = load_plan(p)
            self.assertEqual(back, _SAMPLE_PLAN)

    def test_save_json_by_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "plan.json"
            save_plan(_SAMPLE_PLAN, p)
            self.assertTrue(p.exists())
            content = p.read_text()
            self.assertIn('"target":', content)  # JSON quoted keys
            self.assertEqual(json.loads(content), _SAMPLE_PLAN)
            back = load_plan(p)
            self.assertEqual(back, _SAMPLE_PLAN)

    def test_yaml_ops_use_flow_style(self):
        # Ops dicts should be inline (flow) so each op fits on one line.
        s = plan_to_yaml(_SAMPLE_PLAN)
        # The op line should contain '{op: add_group' inline, not on
        # separate lines.
        self.assertIn("{op: add_group", s)

    def test_save_plan_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "deep" / "nested" / "plan.yaml"
            save_plan(_SAMPLE_PLAN, p)
            self.assertTrue(p.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
