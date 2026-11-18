from __future__ import annotations

import pytest
from pydantic import ValidationError

from synthex.routejson import Route


def test_route_round_trips_legacy_plan():
    plan = {
        "target": "[CH3:1][CH2:2][CH2:3][CH3:4]",
        "steps": [
            {
                "id": "S1",
                "desc": "disconnect the central bond",
                "kind": "key",
                "ops": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
            }
        ],
    }
    route = Route.from_op_plan(plan)
    assert route.format_version == "routejson-1.0"
    assert route.steps[0].kind == "key"
    assert route.to_op_plan() == plan


def test_duplicate_step_ids_are_rejected():
    with pytest.raises(ValidationError, match="identifiers must be unique"):
        Route.model_validate(
            {
                "target": "[CH4:1]",
                "steps": [
                    {"id": "S1", "operations": [{"op": "set_explicit_h", "map_idx": 1, "n": 3}]},
                    {"id": "S1", "operations": [{"op": "set_explicit_h", "map_idx": 1, "n": 2}]},
                ],
            }
        )
