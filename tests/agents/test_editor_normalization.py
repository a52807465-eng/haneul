"""The Editor must survive the step shapes gemini-3.7-flash actually produces.

All three of these failed a live run after the Route Builder had already
succeeded, because the normalization was applied in the Route Builder only and
``Route.from_op_plan`` at submit raised straight out of the turn loop.
"""

from __future__ import annotations

import asyncio

import pytest

from synthex.agents import Editor, EditorRequest
from synthex.agents.base import AgentContext
from synthex.agents.models import Critique
from synthex.llm import ReplayLLMClient
from synthex.routejson import Route

ROUTE = Route(
    target="[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH3:6]",
    steps=[
        {
            "id": "S1",
            "description": "central disconnection",
            "operations": [{"op": "break_bond", "map_a": 3, "map_b": 4}],
        }
    ],
)
CRITIQUE = Critique(
    is_satisfactory=False,
    problematic_steps=[{"step_id": "S1", "severity": "blocking", "feedback": "wrong bond"}],
)


def _call(name, **arguments):
    return {"text": "", "tool_calls": [{"id": f"c-{name}", "name": name, "arguments": arguments}]}


def _edit(old, new):
    return _call("edit_plan", old_string=old, new_string=new)


def _run(responses, max_turns=8):
    client = ReplayLLMClient(responses)
    request = EditorRequest(
        route=ROUTE,
        critique=CRITIQUE,
        strategy="s",
        iteration=1,
        max_iterations=1,
        max_turns=max_turns,
    )
    return asyncio.run(Editor().run(request, AgentContext(llm=client))), client


@pytest.mark.parametrize("kind", ["deprotection", "C-O bond formation", "transform"])
def test_editor_survives_an_out_of_vocabulary_kind(kind):
    """Observed live: three different invented kind values, each fatal."""
    result, client = _run(
        [
            _edit("desc: central disconnection", f"desc: central disconnection\n  kind: {kind}"),
            _call("submit", edit_log="set the kind"),
        ]
    )

    assert result.route.steps[0].kind is None
    assert result.route.steps[0].metadata["reported_kind"] == kind
    assert result.edit_log == "set the kind"


def test_editor_survives_a_split_step_id():
    """Observed live: the Editor split a step into S3a/S3b, which RouteJSON rejects."""
    result, _ = _run(
        [
            _edit("- id: S1", "- id: S1a"),
            _call("submit", edit_log="split the step"),
        ]
    )

    step = result.route.steps[0]
    assert step.id == "S1"
    assert step.metadata["reported_id"] == "S1a"


def test_editor_survives_an_aliased_operation_key():
    result, _ = _run(
        [
            _edit("{op: break_bond, map_a: 3, map_b: 4}", "{type: break_bond, map_a: 3, map_b: 4}"),
            _call("submit", edit_log="renamed the key"),
        ]
    )

    assert result.route.steps[0].operations[0].op == "break_bond"


def test_submit_is_refused_with_feedback_not_raised():
    """A plan that renders but is not valid RouteJSON must re-enter the loop.

    ``render_plan`` ignores the RouteStep schema, so an unrepairable id renders
    fine and used to raise out of Editor.run at submit, failing the whole stage.
    """
    result, client = _run(
        [
            _edit("map_a: 3", "map_a: 99"),  # renders? no -- breaks execution
            _call("submit", edit_log="should be refused"),
            _edit("map_a: 99", "map_a: 3"),
            _call("submit", edit_log="recovered"),
        ]
    )

    assert result.edit_log == "recovered"
    refused = [r for r in result.tool_history if r.name == "submit" and not r.ok]
    assert len(refused) == 1
    assert "submit refused" in refused[0].result


def test_a_valid_kind_is_preserved():
    result, _ = _run(
        [
            _edit("desc: central disconnection", "desc: central disconnection\n  kind: supporting"),
            _call("submit", edit_log="ok"),
        ]
    )

    assert result.route.steps[0].kind == "supporting"
    assert "reported_kind" not in result.route.steps[0].metadata
