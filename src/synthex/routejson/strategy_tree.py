"""Molecule-centric strategy tree.

``op_plan_to_tree`` emits the strict AiZynthFinder shape, which alternates
molecule and reaction nodes::

    mol -> reaction -> mol -> reaction -> mol

That is what the search backend consumes, but it doubles the nesting depth and
splits each step's provenance away from the molecule it produces. This module
emits the flattened shape used by the earlier collaborator reports, where every
node is a molecule and the reaction that makes it is inlined on the same node::

    {
      "molecule": "Cc1ccccc1NNc1ccccc1C",
      "in_stock": false,
      "reaction_smiles": "<precursors>>>product>",
      "mapped_reaction_smiles": "...",
      "strategy": "...",
      "step_rationale": "...",
      "reaction_conditions": "...",
      "ops": [...],
      "children": [<precursor molecules>]
    }

A leaf carries only ``molecule`` and ``in_stock``. Reaction SMILES are written in
the forward (synthetic) direction, precursors on the left, which is the opposite
of the retrosynthetic direction stored on the reaction node.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from synthex.routejson.executor import op_plan_to_tree
from synthex.routejson.model import Route

# Key order is part of the format: it is what makes the file readable.
_REACTION_FIELDS = (
    "reaction_smiles",
    "mapped_reaction_smiles",
    "strategy",
    "step_rationale",
    "reaction_conditions",
    "kind",
    "ops",
)


def _forward(reaction_smiles: str | None) -> str | None:
    """Flip ``product>>precursors`` into the forward ``precursors>>product``."""

    if not reaction_smiles or ">>" not in reaction_smiles:
        return reaction_smiles
    product, _, precursors = reaction_smiles.partition(">>")
    return f"{precursors}>>{product}"


def _flatten(mol_node: dict[str, Any], *, root: bool = False) -> dict[str, Any]:
    node: dict[str, Any] = {
        "molecule": mol_node.get("smiles"),
        "in_stock": bool(mol_node.get("in_stock", False)),
    }
    if root:
        # Atom maps otherwise survive only inside mapped_reaction_smiles, so a
        # route with no steps would lose the mapped target entirely. Keeping it on
        # the root is what lets the tree stand alone as the stored form.
        mapped = mol_node.get("mapped_smiles")
        if mapped:
            node["mapped_molecule"] = mapped
    reactions = [
        child for child in (mol_node.get("children") or []) if child.get("type") == "reaction"
    ]
    if not reactions:
        return node

    reaction = reactions[0]
    metadata = reaction.get("metadata") or {}
    values = {
        "reaction_smiles": _forward(reaction.get("smiles")),
        "mapped_reaction_smiles": _forward(metadata.get("mapped_reaction_smiles")),
        "strategy": metadata.get("reaction_prompt"),
        "step_rationale": metadata.get("step_rationale"),
        "reaction_conditions": metadata.get("reaction_conditions"),
        "kind": metadata.get("kind"),
        "ops": metadata.get("operations"),
    }
    for field in _REACTION_FIELDS:
        if values[field] is not None:
            node[field] = values[field]

    node["children"] = [
        _flatten(child) for child in (reaction.get("children") or []) if child.get("type") == "mol"
    ]
    return node


def op_plan_to_strategy_tree(
    plan: dict[str, Any],
    stock_check: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Return the molecule-centric strategy tree for a lean route-ops plan.

    Equivalent in content to ``op_plan_to_tree`` -- it is built from that tree --
    but every node is a molecule carrying the reaction that produces it, and
    reaction SMILES are written in the forward direction.
    """

    tree = op_plan_to_tree(plan, stock_check=stock_check)
    tree["mapped_smiles"] = plan["target"]
    return _flatten(tree, root=True)


def _to_strict(node: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the alternating mol/reaction shape from a flattened node."""

    mol: dict[str, Any] = {
        "type": "mol",
        "smiles": node.get("molecule"),
        "is_chemical": True,
        "in_stock": bool(node.get("in_stock", False)),
        "children": [],
    }
    children = node.get("children") or []
    if not children:
        return mol

    precursors, _, product = str(node.get("reaction_smiles") or "").rpartition(">>")
    metadata = {
        "operations": node.get("ops") or [],
        "mapped_reaction_smiles": _forward(node.get("mapped_reaction_smiles")),
        "reaction_prompt": node.get("strategy"),
        "step_rationale": node.get("step_rationale"),
        "reaction_conditions": node.get("reaction_conditions"),
        "kind": node.get("kind"),
    }
    mol["children"] = [
        {
            "type": "reaction",
            "smiles": f"{product}>>{precursors}" if precursors else product,
            "is_reaction": True,
            "metadata": {k: v for k, v in metadata.items() if v is not None},
            "children": [_to_strict(child) for child in children],
        }
    ]
    return mol


def strategy_tree_to_op_plan(tree: dict[str, Any]) -> dict[str, Any]:
    """Invert :func:`op_plan_to_strategy_tree`.

    Rebuilds the strict alternating shape and defers to ``tree_to_op_plan``, so
    the step ordering is decided by the same tested traversal rather than a second
    implementation of it.
    """

    from synthex.routejson.tree_to_plan import tree_to_op_plan

    plan = tree_to_op_plan(_to_strict(tree))
    mapped = tree.get("mapped_molecule")
    if mapped:
        plan["target"] = mapped
    return plan


def strategy_tree_to_route(tree: dict[str, Any]) -> Route:
    """Return the RouteJSON document a strategy tree encodes."""

    from synthex.routejson.model import Route as _Route

    return _Route.from_op_plan(strategy_tree_to_op_plan(tree))


__all__ = [
    "op_plan_to_strategy_tree",
    "strategy_tree_to_op_plan",
    "strategy_tree_to_route",
]
