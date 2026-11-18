from __future__ import annotations

import asyncio
import json

import synthex.cli as cli
from synthex.cli import main


def test_reaction_validate_and_apply(tmp_path, capsys):
    path = tmp_path / "reaction.json"
    path.write_text(
        json.dumps(
            {
                "product": "[CH3:1][CH2:2][CH2:3][CH3:4]",
                "operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
            }
        )
    )

    assert main(["reaction", "validate", str(path)]) == 0
    assert "valid reactionjson-1.0" in capsys.readouterr().out
    assert main(["reaction", "apply", str(path)]) == 0
    assert "." in capsys.readouterr().out


def test_route_validate_and_render(tmp_path, capsys):
    path = tmp_path / "route.json"
    path.write_text(
        json.dumps(
            {
                "target": "[CH3:1][CH2:2][CH2:3][CH3:4]",
                "steps": [
                    {
                        "id": "S1",
                        "operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                    }
                ],
            }
        )
    )

    assert main(["route", "validate", str(path)]) == 0
    assert "valid routejson-1.0" in capsys.readouterr().out
    output = tmp_path / "rendered.json"
    assert main(["route", "render", str(path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["route"][0]["id"] == "S1"


def test_invalid_input_has_nonzero_exit(tmp_path, capsys):
    path = tmp_path / "invalid.json"
    path.write_text("{}")
    assert main(["reaction", "validate", str(path)]) == 2
    assert "synthex: error:" in capsys.readouterr().err


def test_batch_reuses_one_event_loop_for_all_targets(tmp_path, monkeypatch):
    class FakePipeline:
        def __init__(self):
            self.loop_ids: list[int] = []

        async def run(self, _smiles, _run_dir, **_kwargs):
            self.loop_ids.append(id(asyncio.get_running_loop()))

    pipeline = FakePipeline()
    monkeypatch.setattr(cli, "_load_live_pipeline", lambda _config, **_kwargs: pipeline)
    targets = tmp_path / "targets.csv"
    targets.write_text("target_id,smiles\none,CC\ntwo,CCC\n")

    assert (
        main(
            [
                "batch",
                str(targets),
                "--config",
                "unused.yml",
                "--output",
                str(tmp_path / "runs"),
            ]
        )
        == 0
    )
    assert len(pipeline.loop_ids) == 2
    assert len(set(pipeline.loop_ids)) == 1


def _one_route_run(tmp_path):
    """Build a real run directory with a reviewed_routes checkpoint."""
    from synthex.agents import Analyst, Critic, Editor, RouteBuilder, StrategyGenerator
    from synthex.config import ManuscriptConfig
    from synthex.llm import ReplayLLMClient
    from synthex.pipeline import PipelineComponents, SynthexPipeline

    responses = [
        {
            "text": json.dumps(
                {
                    "done": False,
                    "step": {
                        "description": "central disconnection",
                        "operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                        "rationale": "splits the chain",
                        "conditions": "NaH, THF",
                        "kind": "key",
                    },
                }
            )
        },
        {"text": '{"done": true}'},
        {
            "text": json.dumps(
                {
                    "is_satisfactory": True,
                    "overall_feedback": "acceptable",
                    "problematic_steps": [
                        {"step_id": "S1", "severity": "suggestion", "feedback": "check selectivity"}
                    ],
                }
            )
        },
    ]
    config = ManuscriptConfig(
        agents={"strategy_count": 1, "route_step_limit": 2, "edit_iteration_limit": 1},
        completion={"enabled": False},
    )
    pipeline = SynthexPipeline(
        config,
        PipelineComponents(
            llm=ReplayLLMClient(responses),
            strategy_generator=StrategyGenerator(),
            route_builder=RouteBuilder(),
            critic=Critic(),
            editor=Editor(),
            analyst=Analyst(),
        ),
    )
    run_dir = tmp_path / "run"
    asyncio.run(
        pipeline.run(
            "CCCC",
            run_dir,
            target_id="T1",
            stages=("strategies", "routes", "reviewed_routes"),
            seed_strategies=["disconnect centrally"],
        )
    )
    return run_dir


def _partial_route_run(tmp_path):
    """Build a real route checkpoint containing one failure and one route."""
    from synthex.agents import Analyst, Critic, Editor, RouteBuilder, StrategyGenerator
    from synthex.config import ManuscriptConfig
    from synthex.llm import ReplayLLMClient
    from synthex.pipeline import PipelineComponents, SynthexPipeline

    step = json.dumps(
        {
            "done": False,
            "step": {
                "description": "central disconnection",
                "operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
            },
        }
    )

    class FirstCallFails:
        def __init__(self):
            self.calls = 0
            self.inner = ReplayLLMClient([{"text": step}, {"text": '{"done": true}'}])

        async def generate(self, request):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("provider exploded")
            return await self.inner.generate(request)

    pipeline = SynthexPipeline(
        ManuscriptConfig(
            agents={"route_step_limit": 2, "route_proposal_attempts": 1},
            completion={"enabled": False},
        ),
        PipelineComponents(
            llm=FirstCallFails(),
            strategy_generator=StrategyGenerator(),
            route_builder=RouteBuilder(),
            critic=Critic(),
            editor=Editor(),
            analyst=Analyst(),
        ),
    )
    run_dir = tmp_path / "partial-run"
    asyncio.run(
        pipeline.run(
            "CCCC",
            run_dir,
            target_id="partial",
            stages=("strategies", "routes"),
            seed_strategies=["unbuildable strategy", "disconnect centrally"],
        )
    )
    return run_dir


def test_show_renders_lean_plan_yaml_not_checkpoint_json(tmp_path, capsys):
    """`show` must emit the scannable plan shape, one op per line."""
    run_dir = _one_route_run(tmp_path)

    assert main(["show", str(run_dir)]) == 0
    out = capsys.readouterr().out

    # target first, ops in flow style, plan field names (not model field names)
    assert out.index("target:") < out.index("steps:")
    assert "- {op: break_bond, map_a: 2, map_b: 3}" in out
    assert "desc: central disconnection" in out
    assert "step_rationale: splits the chain" in out
    assert "reaction_conditions: NaH, THF" in out
    # the strategy and the critic verdict come through as comments
    assert "# strategy: disconnect centrally" in out
    assert "# critic: satisfactory" in out
    assert "[suggestion] S1: check selectivity" in out


def test_show_defaults_to_reviewed_routes_and_accepts_a_stage(tmp_path, capsys):
    run_dir = _one_route_run(tmp_path)

    assert main(["show", str(run_dir)]) == 0
    assert "reviewed_routes[0]" in capsys.readouterr().out

    assert main(["show", str(run_dir), "--stage", "routes"]) == 0
    out = capsys.readouterr().out
    assert "routes[0]" in out
    assert "# critic:" not in out  # the routes stage carries no critique


def test_show_rendered_includes_the_frontier(tmp_path, capsys):
    run_dir = _one_route_run(tmp_path)

    assert main(["show", str(run_dir), "--rendered", "--strip-maps"]) == 0
    out = capsys.readouterr().out
    assert "rendered:" in out
    assert "frontier:" in out
    assert "CCCC>>CC.CC" in out


def test_show_reports_a_missing_or_incomplete_stage(tmp_path, capsys):
    run_dir = _one_route_run(tmp_path)

    assert main(["show", str(run_dir), "--index", "9"]) == 2
    assert "out of range" in capsys.readouterr().err

    assert main(["show", str(tmp_path / "nothing-here")]) == 2
    assert "synthex: error:" in capsys.readouterr().err


def test_show_leaves_the_checkpoints_untouched(tmp_path):
    """The view must never rewrite a digest-linked checkpoint."""
    from synthex.artifacts import ArtifactStore, digest_file

    run_dir = _one_route_run(tmp_path)
    before = {p.name: digest_file(p) for p in sorted(run_dir.glob("*.json"))}

    assert main(["show", str(run_dir), "--rendered"]) == 0

    after = {p.name: digest_file(p) for p in sorted(run_dir.glob("*.json"))}
    assert after == before
    ArtifactStore(run_dir).validate()


def test_show_format_tree_matches_the_collaborator_report_shape(tmp_path, capsys):
    """`--format tree` must emit one node per molecule with its reaction inlined."""
    run_dir = _one_route_run(tmp_path)

    assert main(["show", str(run_dir), "--index", "0", "--format", "tree"]) == 0
    tree = json.loads(capsys.readouterr().out)

    assert list(tree) == [
        "molecule",
        "in_stock",
        "mapped_molecule",
        "reaction_smiles",
        "mapped_reaction_smiles",
        "strategy",
        "step_rationale",
        "reaction_conditions",
        "kind",
        "ops",
        "children",
    ]
    assert tree["molecule"] == "CCCC"
    # forward direction: precursors on the left, product on the right
    assert tree["reaction_smiles"] == "CC.CC>>CCCC"
    assert tree["mapped_reaction_smiles"].endswith(">>[CH3:1][CH2:2][CH2:3][CH3:4]")
    assert tree["strategy"] == "central disconnection"
    assert tree["ops"] == [{"op": "break_bond", "map_a": 2, "map_b": 3}]
    # leaves carry only the molecule and its stock flag
    assert [sorted(child) for child in tree["children"]] == [
        ["in_stock", "molecule"],
        ["in_stock", "molecule"],
    ]


def test_show_format_tree_returns_a_list_without_an_index(tmp_path, capsys):
    run_dir = _one_route_run(tmp_path)

    assert main(["show", str(run_dir), "--format", "tree"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert isinstance(document, list)
    assert [node["molecule"] for node in document] == ["CCCC"]


def test_show_preserves_failed_items_alongside_readable_plans(tmp_path, capsys):
    run_dir = _partial_route_run(tmp_path)

    assert main(["show", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "# routes[0]" in out
    assert "# strategy: unbuildable strategy" in out
    assert "# status: failed" in out
    assert "# failure: RuntimeError: provider exploded" in out
    assert "status: failed" in out
    assert "# routes[1]" in out
    assert "target:" in out


def test_show_format_tree_preserves_failed_items(tmp_path, capsys):
    run_dir = _partial_route_run(tmp_path)

    assert main(["show", str(run_dir), "--format", "tree"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document[0] == {
        "failure": {"category": "RuntimeError", "message": "provider exploded"},
        "status": "failed",
        "strategy": "unbuildable strategy",
    }
    assert document[1]["molecule"] == "CCCC"

    assert main(["show", str(run_dir), "--format", "tree", "--index", "0"]) == 0
    failure = json.loads(capsys.readouterr().out)
    assert failure["status"] == "failed"
    assert "molecule" not in failure
