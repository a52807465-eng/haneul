from __future__ import annotations

import json
from pathlib import Path

from synthex.artifacts import run_manifest_schema
from synthex.reactionjson import Reaction, reaction_schema
from synthex.routejson import Route, route_schema

ROOT = Path(__file__).parents[1]


def test_checked_in_schemas_match_models():
    assert (
        json.loads((ROOT / "schemas/reactionjson-1.0.schema.json").read_text()) == reaction_schema()
    )
    assert json.loads((ROOT / "schemas/routejson-1.0.schema.json").read_text()) == route_schema()
    assert (
        json.loads((ROOT / "schemas/run-manifest-1.0.schema.json").read_text())
        == run_manifest_schema()
    )


def test_examples_validate():
    Reaction.model_validate_json((ROOT / "examples/reactionjson/bond-break.json").read_text())
    Route.model_validate_json((ROOT / "examples/routejson/two-step-route.json").read_text())
