"""Enrich a lean route-ops plan with derived fields.

``tree_to_op_plan`` emits a lean plan (just ``target`` + ``steps[]`` w/ ops
+ descriptions). For interactive viewing (Route Editing agent, debug, plot)
we need:

* ``steps[].rendered.frontier`` — the current mol set after each step.
* ``steps[].rendered.solved`` — true when every frontier mol is in stock.
* top-level ``route[]`` — quick unmapped reaction-SMILES summary, one
  entry per step.

These are deterministically derivable from ``target`` + ``steps[].ops`` by
re-running the executor, so we don't persist them in the on-disk plan.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from rdkit import Chem

from synthex.reactionjson.executor import execute_ops
from synthex.routejson.executor import (
    _assign_offset_maps_inplace,
    _required_map_ids,
)

_MAP_RE = re.compile(r":(\d+)\]")


def _maps_in(smi: str) -> set[int]:
    return {int(m) for m in _MAP_RE.findall(smi)}


def _strip_maps_canonical(mapped_smi: str) -> str:
    mol = Chem.MolFromSmiles(mapped_smi)
    if mol is None:
        return mapped_smi
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def _resolve_up_to(plan: dict[str, Any], up_to: int | str | None) -> int:
    """Translate ``up_to`` into a 1-based step count to render."""
    steps = plan.get("steps") or []
    if up_to is None:
        return len(steps)
    if isinstance(up_to, int):
        if up_to < 1 or up_to > len(steps):
            raise ValueError(f"up_to={up_to} out of range [1, {len(steps)}]")
        return up_to
    # String: match step id.
    for i, s in enumerate(steps, start=1):
        if s.get("id") == up_to:
            return i
    raise ValueError(f"up_to={up_to!r} matches no step id in plan")


def render_plan(
    plan: dict[str, Any],
    stock_check: Callable[[str], bool] | None = None,
    strip_frontier_maps: bool = False,
    up_to: int | str | None = None,
) -> dict[str, Any]:
    """Return a deep copy of ``plan`` with ``rendered`` + ``route`` filled in.

    Does not mutate the input. Re-executes ``steps[].ops`` against
    ``plan['target']`` to derive each step's frontier and the bottom route
    summary.

    Parameters
    ----------
    plan
        Lean plan as produced by ``tree_to_op_plan``.
    stock_check
        Optional ``(canonical_smiles) -> bool`` for the ``in_stock`` flag.
        Default: nothing in stock.
    strip_frontier_maps
        If True, ``rendered.frontier[].smi`` is unmapped canonical SMILES.
        Default keeps mapped form so later steps' ops remain addressable.
    up_to
        Render only the first N steps. ``None`` (default) renders all.
        ``int`` is the 1-based step number; ``str`` is the step id
        ("S3"). Steps past the cutoff are left in the plan as-is (no
        ``rendered`` key); ``route[]`` is truncated accordingly.
    """
    in_stock_fn = stock_check or (lambda _smi: False)
    target_mapped = plan["target"]
    used_maps: set[int] = set(_maps_in(target_mapped))
    cutoff = _resolve_up_to(plan, up_to)

    # Frontier entry: {"mapped": str, "plain": str, "in_stock": bool}.
    frontier: list[dict[str, Any]] = [
        {
            "mapped": target_mapped,
            "plain": _strip_maps_canonical(target_mapped),
            "in_stock": in_stock_fn(_strip_maps_canonical(target_mapped)),
        }
    ]

    out = deepcopy(plan)
    out_route: list[dict[str, str]] = []
    # Per-rendered-step frontier snapshots (lists of mol dicts that we
    # mutate later to attach the is_leaf flag once we know the final set).
    frontier_snapshots: list[list[dict[str, Any]]] = []

    for step_index, step in enumerate(out["steps"]):
        if step_index >= cutoff:
            # Clear any stale rendered field on steps past the cutoff.
            step.pop("rendered", None)
            continue
        ops = step.get("ops") or []
        if any(o.get("op") == "__interpolated__" for o in ops):
            raise ValueError(
                f"step {step.get('id')}: ops contain '__interpolated__' "
                "placeholder; plan cannot be rendered deterministically"
            )

        req_maps = _required_map_ids(ops)
        product_idx = None
        for i, mol in enumerate(frontier):
            if req_maps.issubset(_maps_in(mol["mapped"])):
                product_idx = i
                break
        if product_idx is None:
            raise ValueError(
                f"step {step.get('id')}: op map ids {sorted(req_maps)} not in "
                f"any frontier mol (frontier size {len(frontier)})"
            )
        product = frontier.pop(product_idx)

        reactants_mapped, err = execute_ops(ops, product["mapped"])
        if err:
            raise ValueError(f"step {step.get('id')}: execute_ops failed: {err}")

        rmol = Chem.MolFromSmiles(reactants_mapped)
        if rmol is None:
            raise ValueError(f"step {step.get('id')}: unparseable reactants {reactants_mapped!r}")
        frag_mols = Chem.GetMolFrags(rmol, asMols=True, sanitizeFrags=True)
        for fm in frag_mols:
            _assign_offset_maps_inplace(fm, used_maps)

        new_mols: list[dict[str, Any]] = []
        for fm in frag_mols:
            mapped = Chem.MolToSmiles(fm)
            plain = _strip_maps_canonical(mapped)
            new_mols.append(
                {
                    "mapped": mapped,
                    "plain": plain,
                    "in_stock": in_stock_fn(plain),
                }
            )
        frontier.extend(new_mols)

        # Build rendered entries; is_leaf is filled in below after we know
        # the final frontier's mol set.
        rendered_frontier = [
            {
                "smi": (m["plain"] if strip_frontier_maps else m["mapped"]),
                "in_stock": m["in_stock"],
                "is_leaf": False,  # placeholder, set after the loop
                # internal: used to match against final frontier; stripped at the end.
                "_mapped": m["mapped"],
            }
            for m in frontier
        ]
        frontier_snapshots.append(rendered_frontier)
        step["rendered"] = {
            "frontier": rendered_frontier,
            "solved": bool(frontier) and all(m["in_stock"] for m in frontier),
        }

        reactants_plain = ".".join(m["plain"] for m in new_mols)
        out_route.append(
            {
                "id": step["id"],
                "smiles": f"{product['plain']}>>{reactants_plain}",
            }
        )

    # Identify route leaves = mols that persist in the final frontier (i.e.
    # were never consumed as a product by any subsequent step). Match by
    # mapped-SMILES identity since maps are stable across steps.
    leaf_mapped_set = {m["mapped"] for m in frontier} if frontier else set()
    for snapshot in frontier_snapshots:
        for entry in snapshot:
            entry["is_leaf"] = entry.pop("_mapped") in leaf_mapped_set

    out["route"] = out_route
    return out
