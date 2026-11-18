"""The molecule-centric strategy tree must carry the same content as the
AiZynthFinder-shaped tree, only flattened and written forward."""

from __future__ import annotations

from synthex.routejson import Route, op_plan_to_strategy_tree, op_plan_to_tree

PLAN = Route(
    target="[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH3:6]",
    steps=[
        {
            "id": "S1",
            "description": "central disconnection",
            "operations": [{"op": "break_bond", "map_a": 3, "map_b": 4}],
            "rationale": "splits the chain in half",
            "conditions": "Na, ether",
            "kind": "key",
        }
    ],
).to_op_plan()


def _molecules(node):
    yield node["molecule"]
    for child in node.get("children") or []:
        yield from _molecules(child)


def test_every_node_is_a_molecule():
    """No reaction layer: the alternating mol/reaction nesting is collapsed."""
    tree = op_plan_to_strategy_tree(PLAN)

    def visit(node):
        assert "molecule" in node
        assert node.get("type") is None
        for child in node.get("children") or []:
            visit(child)

    visit(tree)


def test_reaction_smiles_is_written_forward():
    tree = op_plan_to_strategy_tree(PLAN)
    reaction = op_plan_to_tree(PLAN)["children"][0]

    retro_product, _, retro_precursors = reaction["smiles"].partition(">>")
    assert tree["reaction_smiles"] == f"{retro_precursors}>>{retro_product}"
    assert tree["reaction_smiles"].endswith(">>CCCCCC")


def test_step_provenance_is_inlined_on_the_molecule():
    tree = op_plan_to_strategy_tree(PLAN)

    assert tree["strategy"] == "central disconnection"
    assert tree["step_rationale"] == "splits the chain in half"
    assert tree["reaction_conditions"] == "Na, ether"
    assert tree["ops"] == [{"op": "break_bond", "map_a": 3, "map_b": 4}]


def test_leaves_carry_only_the_molecule_and_stock_flag():
    tree = op_plan_to_strategy_tree(PLAN)

    for child in tree["children"]:
        assert sorted(child) == ["in_stock", "molecule"]


def test_molecule_set_is_unchanged_from_the_aizynthfinder_tree():
    """Flattening must not add, drop or reorder molecules."""
    flat = list(_molecules(op_plan_to_strategy_tree(PLAN)))

    strict = []

    def visit(node):
        if node.get("type") == "mol":
            strict.append(node["smiles"])
        for child in node.get("children") or []:
            visit(child)

    visit(op_plan_to_tree(PLAN))
    assert flat == strict


def test_stock_check_is_honoured():
    tree = op_plan_to_strategy_tree(PLAN, stock_check=lambda smiles: smiles == "CCC")

    assert tree["in_stock"] is False
    assert all(child["in_stock"] for child in tree["children"])


def test_a_target_with_no_steps_keeps_its_atom_mapping():
    """The root carries mapped_molecule, or a zero-step route loses its maps.

    Atom maps otherwise survive only inside mapped_reaction_smiles, which a route
    with no reactions does not have, so the mapped target would be unrecoverable
    and the tree could not stand alone as the stored form.
    """
    plan = Route(target="[CH4:1]").to_op_plan()

    assert op_plan_to_strategy_tree(plan) == {
        "molecule": "C",
        "in_stock": False,
        "mapped_molecule": "[CH4:1]",
    }


def test_the_tree_round_trips_back_to_the_same_route():
    """The tree is only usable as the stored form if nothing is lost."""
    from synthex.routejson.strategy_tree import strategy_tree_to_route

    route = Route.from_op_plan(PLAN)
    back = strategy_tree_to_route(op_plan_to_strategy_tree(PLAN))

    assert back.target == route.target
    assert len(back.steps) == len(route.steps)
    for before, after in zip(route.steps, back.steps, strict=True):
        assert after.id == before.id
        assert after.description == before.description
        assert after.rationale == before.rationale
        assert after.conditions == before.conditions
        assert after.kind == before.kind
        assert [op.model_dump(exclude_none=True) for op in after.operations] == [
            op.model_dump(exclude_none=True) for op in before.operations
        ]


def test_a_zero_step_route_round_trips():
    from synthex.routejson.strategy_tree import strategy_tree_to_route

    plan = Route(target="[CH4:1]").to_op_plan()
    assert strategy_tree_to_route(op_plan_to_strategy_tree(plan)).target == "[CH4:1]"
