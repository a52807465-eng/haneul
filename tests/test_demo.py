from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_offline_reviewer_demo(tmp_path):
    output = tmp_path / "demo-output"
    result = subprocess.run(
        [sys.executable, str(ROOT / "demo/run_demo.py"), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    expected = json.loads((ROOT / "demo/expected-summary.json").read_text())
    assert json.loads((output / "demo-summary.json").read_text()) == expected
    assert "SynthEx offline demo: PASS" in result.stdout
