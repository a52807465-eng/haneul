"""Convert a route-tree dict into a (lean) route-ops plan.

Input: the dict returned by ``ReactionTree.to_dict()`` (root mol with nested
``children`` of alternating reaction / mol nodes).

Output: lean plan — only authored content. Detailed ``rendered.frontier``
per step and the bottom ``route[]`` summary are derived by ``render_plan``
on demand, not stored on disk.

    {
      "target": "<mapped SMILES of root mol>",
      "steps": [
        {
          "id": "S1",
          "desc": "<short LLM reaction_prompt>",
          "step_rationale": "<strategic reasoning, longer-form>",   # if present
          "reaction_conditions": "<typical forward conditions>",    # if present
          "ops": [...],
        },
        ...
      ],
    }

``desc`` is the short ``reaction_prompt`` (one sentence). ``step_rationale``
and ``reaction_conditions`` are richer fields emitted only when present in
the reaction metadata. A short ``type`` label is intentionally omitted (LLM
phrasing is too inconsistent to extract reliably).

Use ``synthex.routejson.render_plan`` to add ``rendered.frontier``
per step + the bottom ``route[]`` summary when needed (e.g. by the Route
Editing agent for visual inspection).
"""

from __future__ import annotations

import re
from typing import Any

from synthex.routejson.propagate import propagate_map_ids

_MAP_RE = re.compile(r":(\d+)\]")


def _extract_desc(meta: dict[str, Any]) -> str:
    """Pull the short reaction description out of metadata.

    Prefer ``strategy`` — the text that produced the accepted ops. Older
    runs only carry ``reaction_prompt``, the planner's hint, which
    describes a different reaction when the accepted ops came from a
    fallback round (``reaction_round > 0``); cache-hit reconstructions
    surface it as ``original_strategy``.
    """
    return (
        meta.get("strategy") or meta.get("reaction_prompt") or meta.get("original_strategy") or ""
    )


def tree_to_op_plan(route_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a route-tree dict into a lean route-ops plan.

    Walks the existing nested ``mol/reaction`` tree, collects authored
    content per step. Does NOT execute ops or derive a frontier — call
    ``render_plan`` for that.
    """
    if route_dict.get("type") != "mol":
        raise ValueError("route_dict must be a route tree rooted at a 'mol' node")

    # Collect reactions in DFS pre-order so each step's ops act on the state
    # left by the previous step.
    step_records: list[dict[str, Any]] = []

    def walk(mol_node: dict[str, Any]) -> None:
        for rxn_node in mol_node.get("children") or []:
            step_records.append(rxn_node)
            for child_mol in rxn_node.get("children") or []:
                walk(child_mol)

    walk(route_dict)

    if not step_records:
        return {
            "target": route_dict.get("smiles", ""),
            "steps": [],
        }

    # Target = LHS of first step's mapped_reaction_smiles (preserves the
    # global atom-map space the patched code-ops policy produced).
    first_meta = step_records[0].get("metadata", {}) or {}
    first_mrs = first_meta.get("mapped_reaction_smiles", "")
    if ">>" in first_mrs:
        target_mapped = first_mrs.split(">>", 1)[0]
    else:
        target_mapped = first_meta.get("mapped_product_smiles") or route_dict.get("smiles", "")

    plan_steps: list[dict[str, Any]] = []
    prop_input: list[dict[str, Any]] = []
    for i, rxn in enumerate(step_records, start=1):
        meta = rxn.get("metadata", {}) or {}
        ops = meta.get("operations")
        if ops is None:
            # Legacy / cache-hit-pre-patch fallback: emit a placeholder so
            # callers can detect and decide (interpolator or re-run).
            ops = [{"op": "__interpolated__", "reason": "no operations in metadata"}]

        step_dict: dict[str, Any] = {
            "id": f"S{i}",
            "desc": _extract_desc(meta),
        }
        if meta.get("step_rationale"):
            step_dict["step_rationale"] = meta["step_rationale"]
        if meta.get("reaction_conditions"):
            step_dict["reaction_conditions"] = meta["reaction_conditions"]
        if meta.get("kind"):
            step_dict["kind"] = meta["kind"]
        step_dict["ops"] = ops
        plan_steps.append(step_dict)
        prop_input.append(
            {
                "ops": ops,
                "mapped_reaction_smiles": meta.get("mapped_reaction_smiles", ""),
            }
        )

    # The search policy reuses freed atom-map ids; the renderer never does.
    # Rewrite each step's op references into the renderer's numbering so the
    # plan renders deterministically and its maps stay globally unique. Falls
    # back to the verbatim ops if the route can't be replayed.
    rewritten = propagate_map_ids(target_mapped, prop_input)
    if rewritten is not None:
        for step_dict, ops_global in zip(plan_steps, rewritten, strict=True):
            step_dict["ops"] = ops_global

    return {
        "target": target_mapped,
        "steps": plan_steps,
    }
