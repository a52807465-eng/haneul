from __future__ import annotations

import asyncio
import json

import pytest

from synthex.agents import (
    AgentContext,
    Analyst,
    AnalystRequest,
    Critic,
    CriticRequest,
    Editor,
    EditorRequest,
    RouteBuilder,
    RouteBuildRequest,
    StrategyGenerator,
    StrategyRequest,
)
from synthex.llm import LLMResponse, ReplayLLMClient, ToolCall
from synthex.routejson import Route


def _run(awaitable):
    return asyncio.run(awaitable)


def _one_step_route() -> Route:
    return Route.model_validate(
        {
            "target": "[CH3:1][CH2:2][CH2:3][CH3:4]",
            "steps": [
                {
                    "id": "S1",
                    "description": "Central disconnection",
                    "operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                }
            ],
        }
    )


def test_strategy_generator_replay_uses_neutral_client():
    client = ReplayLLMClient([{"text": '{"strategies": ["A", "B", "C"]}'}])
    result = _run(
        StrategyGenerator().run(StrategyRequest(target="CCCC", count=2), AgentContext(llm=client))
    )
    assert result.strategies == ["A", "B"]
    assert client.requests[0].temperature == 0.1
    assert client.requests[0].thinking_level is None
    assert client.requests[0].response_schema is not None


def test_route_builder_replays_graph_edits_and_stops_early():
    client = ReplayLLMClient(
        [
            {
                "text": json.dumps(
                    {
                        "done": False,
                        "step": {
                            "id": "ignored",
                            "description": "Central disconnection",
                            "operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                            "kind": "key",
                        },
                    }
                )
            },
            {"text": '{"done": true, "reason": "reasonable precursors"}'},
        ]
    )
    result = _run(
        RouteBuilder().run(
            RouteBuildRequest(target="CCCC", strategy="disconnect centrally"),
            AgentContext(llm=client),
        )
    )
    assert result.termination == "model_finished"
    assert result.route.steps[0].id == "S1"
    assert result.route.steps[0].kind == "key"
    assert len(client.requests) == 2
    assert client.requests[0].temperature == 0.3
    assert client.requests[0].thinking_level == "medium"


def test_route_builder_retries_an_operation_that_does_not_execute():
    client = ReplayLLMClient(
        [
            {
                "text": json.dumps(
                    {
                        "done": False,
                        "step": {
                            "id": "S1",
                            "operations": [{"op": "break_bond", "map_a": 2, "map_b": 99}],
                        },
                    }
                )
            },
            {
                "text": json.dumps(
                    {
                        "done": False,
                        "step": {
                            "id": "S1",
                            "operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                        },
                    }
                )
            },
            {"text": '{"done": true}'},
        ]
    )
    result = _run(
        RouteBuilder().run(
            RouteBuildRequest(target="CCCC", strategy="disconnect centrally"),
            AgentContext(llm=client),
        )
    )
    assert len(result.route.steps) == 1
    retry_payload = json.loads(client.requests[1].messages[0].content)
    assert "map ids" in retry_payload["previous_error"]


def test_critic_accepts_tagged_source_prompt_response():
    response = """<feedback>{
      "is_satisfactory": false,
      "overall_feedback": "one blocking issue",
      "problematic_steps": [{"step_id": "S1", "severity": "blocking", "feedback": "risk"}]
    }</feedback>"""
    client = ReplayLLMClient([{"text": response}])
    result = _run(Critic().run(CriticRequest(route=_one_step_route()), AgentContext(llm=client)))
    assert not result.is_resolved
    assert result.blocking_steps[0].step_id == "S1"
    assert client.requests[0].temperature == 0.1
    assert client.requests[0].thinking_level == "medium"


def test_editor_tools_are_provider_neutral_and_submit_renderable_route():
    client = ReplayLLMClient(
        [
            LLMResponse(tool_calls=(ToolCall(id="1", name="read_plan", arguments={}),)),
            LLMResponse(
                tool_calls=(
                    ToolCall(
                        id="2",
                        name="submit",
                        arguments={"edit_log": "No structural edit required"},
                    ),
                )
            ),
        ]
    )
    critique = {
        "is_satisfactory": True,
        "overall_feedback": "acceptable",
        "problematic_steps": [],
    }
    result = _run(
        Editor().run(
            EditorRequest(
                route=_one_step_route(), critique=critique, strategy="central disconnection"
            ),
            AgentContext(llm=client),
        )
    )
    assert result.route == _one_step_route()
    assert [item.name for item in result.tool_history] == ["read_plan", "submit"]
    assert all(tool.__class__.__module__ == "synthex.llm" for tool in client.requests[0].tools)
    assert client.requests[0].temperature == 0.3
    assert client.requests[0].thinking_level == "medium"


def test_editor_can_read_history_and_revert_to_a_prior_snapshot():
    client = ReplayLLMClient(
        [
            LLMResponse(tool_calls=(ToolCall(id="1", name="read_edit_history", arguments={}),)),
            LLMResponse(
                tool_calls=(ToolCall(id="2", name="revert_to", arguments={"iteration": 0}),)
            ),
            LLMResponse(
                tool_calls=(ToolCall(id="3", name="submit", arguments={"edit_log": "reverted"}),)
            ),
        ]
    )
    route = _one_step_route()
    result = _run(
        Editor().run(
            EditorRequest(
                route=route,
                critique={"problematic_steps": []},
                strategy="central disconnection",
                iteration=2,
                max_iterations=6,
                history=[{"iteration": 1, "edit_log": "first pass"}],
                snapshots={0: route},
            ),
            AgentContext(llm=client),
        )
    )
    assert "first pass" in result.tool_history[0].result
    assert result.tool_history[1].result == "OK: reverted to route snapshot 0."
    assert result.edit_log == "reverted"


def test_analyst_accepts_typed_replay_result():
    client = ReplayLLMClient(
        [
            {
                "text": json.dumps(
                    {
                        "feasibility_category": "acceptable",
                        "summary": "Plausible route",
                        "key_steps": ["Central disconnection"],
                        "risks": ["selectivity"],
                        "details": {},
                    }
                )
            }
        ]
    )
    result = _run(Analyst().run(AnalystRequest(route=_one_step_route()), AgentContext(llm=client)))
    assert result.feasibility_category == "acceptable"
    assert result.risks == ["selectivity"]
    assert client.requests[0].temperature == 0.3
    assert client.requests[0].thinking_level is None


@pytest.mark.parametrize("reported", ["transform", "forward", "KEY", "", 7, ["key"]])
def test_route_builder_demotes_an_out_of_vocabulary_step_kind(reported):
    """An unusable ``kind`` label must not discard a valid disconnection.

    ``RouteBuilderTurn.step`` is an untyped object, so the response schema cannot
    constrain ``kind`` and a model is free to invent a label. Observed live from
    gemini-3.7-flash: "transform" and "forward", each of which used to exhaust the
    whole proposal budget and fail the routes stage.
    """

    client = ReplayLLMClient(
        [
            {
                "text": json.dumps(
                    {
                        "done": False,
                        "step": {
                            "description": "Central disconnection",
                            "operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                            "kind": reported,
                        },
                    }
                )
            },
            {"text": '{"done": true}'},
        ]
    )
    result = _run(
        RouteBuilder().run(
            RouteBuildRequest(target="CCCC", strategy="disconnect centrally"),
            AgentContext(llm=client),
        )
    )

    step = result.route.steps[0]
    assert step.kind is None
    assert step.metadata["reported_kind"] == reported
    # One proposal, not a burned retry budget.
    assert len(client.requests) == 2


@pytest.mark.parametrize("reported", ["key", "supporting"])
def test_route_builder_keeps_a_valid_step_kind(reported):
    client = ReplayLLMClient(
        [
            {
                "text": json.dumps(
                    {
                        "done": False,
                        "step": {
                            "description": "Central disconnection",
                            "operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                            "kind": reported,
                        },
                    }
                )
            },
            {"text": '{"done": true}'},
        ]
    )
    result = _run(
        RouteBuilder().run(
            RouteBuildRequest(target="CCCC", strategy="disconnect centrally"),
            AgentContext(llm=client),
        )
    )

    assert result.route.steps[0].kind == reported
    assert "reported_kind" not in result.route.steps[0].metadata


@pytest.mark.parametrize("alias", ["type", "name", "operation"])
def test_route_builder_accepts_an_aliased_operation_name_key(alias):
    """An operation naming itself under `type` must not fail the step.

    ``Operation`` is a discriminated union tagged on ``op``, and the prompt used to
    describe the vocabulary in function-call syntax without ever showing the JSON
    encoding. gemini-3.7-flash emitted ``{"type": "break_bond", ...}`` and burned
    the whole proposal budget on an opaque union_tag_not_found for every entry.
    """

    client = ReplayLLMClient(
        [
            {
                "text": json.dumps(
                    {
                        "done": False,
                        "step": {
                            "description": "Central disconnection",
                            "operations": [{alias: "break_bond", "map_a": 2, "map_b": 3}],
                        },
                    }
                )
            },
            {"text": '{"done": true}'},
        ]
    )
    result = _run(
        RouteBuilder().run(
            RouteBuildRequest(target="CCCC", strategy="disconnect centrally"),
            AgentContext(llm=client),
        )
    )

    operation = result.route.steps[0].operations[0]
    assert operation.op == "break_bond"
    assert (operation.map_a, operation.map_b) == (2, 3)
    assert len(client.requests) == 2  # one proposal, not a burned budget


def test_route_builder_does_not_rewrite_an_operation_that_already_has_op():
    """An explicit `op` wins; a stray `type` alongside it must still be rejected."""

    client = ReplayLLMClient(
        [
            {
                "text": json.dumps(
                    {
                        "done": False,
                        "step": {
                            "description": "Central disconnection",
                            "operations": [
                                {"op": "break_bond", "type": "nonsense", "map_a": 2, "map_b": 3}
                            ],
                        },
                    }
                )
            }
        ]
        * 3
    )
    with pytest.raises(ValueError, match="exhausted 3 proposals"):
        _run(
            RouteBuilder().run(
                RouteBuildRequest(target="CCCC", strategy="disconnect centrally"),
                AgentContext(llm=client),
            )
        )


def test_route_builder_prompt_shows_the_json_encoding_of_an_operation():
    """The prompt must show `op` as a JSON key, not just function-call syntax."""
    from synthex.prompts.route_builder_v1 import route_builder_prompt

    assert '{"op": "break_bond", "map_a": <int>, "map_b": <int>}' in route_builder_prompt
    for name in (
        "add_bond",
        "change_bond_order",
        "change_atom",
        "set_explicit_h",
        "add_group",
        "remove_group",
        "invert_stereocenter",
        "clear_stereocenter",
        "set_bond_stereo",
    ):
        assert f'{{"op": "{name}"' in route_builder_prompt


def test_critic_is_given_the_rdkit_atom_accounting():
    """The Critic prompt must carry a computed accounting block, one entry per step."""
    from synthex.agents.critic import _accounting_block
    from synthex.routejson import render_plan

    route = _one_step_route()
    rendered = render_plan(route.to_op_plan())
    block = _accounting_block(rendered)

    assert "Step S1:" in block
    assert "MISSING from precursors" in block
    assert "Stereocentres" in block

    client = ReplayLLMClient(
        [
            {
                "text": json.dumps(
                    {
                        "is_satisfactory": True,
                        "overall_feedback": "fine",
                        "balance_check": "SURPLUS none, MISSING none. PASS",
                        "problematic_steps": [],
                    }
                )
            }
        ]
    )
    result = _run(Critic().run(CriticRequest(route=route), AgentContext(llm=client)))

    prompt = client.requests[0].messages[0].content
    assert "<atom_stereo_accounting>" in prompt
    assert "{{ATOM_ACCOUNTING}}" not in prompt
    assert "MISSING from precursors" in prompt
    assert result.balance_check == "SURPLUS none, MISSING none. PASS"


def test_critique_without_a_balance_check_still_validates():
    """The field is additive: an older critique payload must keep working."""
    from synthex.agents.models import Critique

    critique = Critique.model_validate({"is_satisfactory": True, "problematic_steps": []})
    assert critique.balance_check is None


def test_route_builder_prompt_carries_the_ported_guidance():
    """Finding 1 from the synthelite port audit: composition patterns and rules."""
    from synthex.prompts.route_builder_v1 import route_builder_prompt

    assert "COMPOSITION PATTERNS" in route_builder_prompt
    assert "Retro-Diels-Alder" in route_builder_prompt
    assert "Retro-aldol" in route_builder_prompt
    assert "Retro-acetalization" in route_builder_prompt
    # remove_group's argument order is the silent-failure trap
    assert "Reversing the two arguments" in route_builder_prompt
    assert "STEREOCHEMISTRY PRESERVATION (CRITICAL)" in route_builder_prompt
    # conditions must be abstract, per the newer synthelite guidance
    assert "Do not name specific reagents or" in route_builder_prompt


def test_critic_recovers_from_an_unreadable_first_reply():
    """One malformed reply used to fail the stage after the route spend was gone."""
    good = json.dumps({"is_satisfactory": True, "overall_feedback": "ok", "problematic_steps": []})
    client = ReplayLLMClient(
        [
            # JSON with a // comment -- which the prompt's own example invites
            {
                "text": '<feedback>\n{"is_satisfactory": true,\n // note\n "problematic_steps": []}\n</feedback>'
            },
            {"text": f"<feedback>{good}</feedback>"},
        ]
    )
    result = _run(Critic().run(CriticRequest(route=_one_step_route()), AgentContext(llm=client)))

    assert result.is_satisfactory is True
    assert len(client.requests) == 2
    # the retry must tell the model what was wrong
    retry = client.requests[1].messages[-1].content
    assert "could not be read" in retry
    assert "no comments inside the JSON" in retry


def test_critic_gives_up_after_its_attempt_budget():
    client = ReplayLLMClient([{"text": "no json here at all"}] * 3)
    with pytest.raises(ValueError, match="no readable feedback in 3 attempts"):
        _run(Critic().run(CriticRequest(route=_one_step_route()), AgentContext(llm=client)))
    assert len(client.requests) == 3


def test_analyst_recovers_from_a_missing_tag():
    tags = {
        "route_summary": {"summary": "s"},
        "key_steps": {"key_steps": []},
        "stereoselectivity": {},
        "handover_point": {},
        "convergence": {},
        "redundant_reactions": {},
        "ops_completeness": {},
        "reaction_conditions": {},
        "starting_materials": {},
        "feasibility_verdict": {"overall_rating": "acceptable", "one_line_summary": "ok"},
        "asymmetric_catalysis": {},
        "cascade_reactions": {},
        "route_score": {},
    }
    complete = "".join(f"<{k}>{json.dumps(v)}</{k}>" for k, v in tags.items())
    truncated = complete[: complete.index("<feasibility_verdict>")]

    client = ReplayLLMClient([{"text": truncated}, {"text": complete}])
    result = _run(Analyst().run(AnalystRequest(route=_one_step_route()), AgentContext(llm=client)))

    assert result.feasibility_category == "acceptable"
    assert len(client.requests) == 2
    assert "could not be read" in client.requests[1].messages[-1].content


def test_route_builder_retry_feedback_names_the_nearest_operation():
    """An invented operation name must come back with an actionable suggestion."""
    client = ReplayLLMClient(
        [
            {
                "text": json.dumps(
                    {
                        "done": False,
                        "step": {
                            "description": "d",
                            "operations": [
                                {"op": "attach_fragment", "map_idx": 2, "fragment_smiles": "Br"}
                            ],
                        },
                    }
                )
            },
            {
                "text": json.dumps(
                    {
                        "done": False,
                        "step": {
                            "description": "d",
                            "operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                        },
                    }
                )
            },
            {"text": '{"done": true}'},
        ]
    )
    result = _run(
        RouteBuilder().run(
            RouteBuildRequest(target="CCCC", strategy="disconnect centrally"),
            AgentContext(llm=client),
        )
    )

    assert result.route.steps[0].operations[0].op == "break_bond"
    feedback = json.loads(client.requests[1].messages[0].content)["previous_error"]
    assert "'attach_fragment' is not an operation" in feedback
    assert "Did you mean 'add_group'?" in feedback
    assert "pydantic.dev" not in feedback  # no URL, no blob
