"""Materialise a route-ops plan back into a nested mol/reaction tree.

The reverse of ``tree_to_op_plan``: take a plan dict, walk its steps in
order, apply each step's ops to the appropriate frontier molecule, and
build the same nested tree shape ``ReactionTree.to_dict()`` produces so
the existing scorer / validator / RouteCollection consume it unchanged.

Map propagation relies on the patches in ``synthelite/utils/atomic_ops/``
+ ``synthelite/utils/atomic_op_executor.py`` (``preserve_maps=True``).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from rdkit import Chem

from synthex.reactionjson.executor import execute_ops

_MAP_RE = re.compile(r":(\d+)\]")
# Atom maps inside a fragment SMILES (e.g. the ``:98`` in ``*[S:98]CC[S:99]``).
_FRAG_MAP_RE = re.compile(r":(\d+)")


def _maps_in(smi: str) -> set[int]:
    return {int(m) for m in _MAP_RE.findall(smi)}


def _strip_maps_to_canonical(mapped_smi: str) -> str:
    mol = Chem.MolFromSmiles(mapped_smi)
    if mol is None:
        return mapped_smi
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


# Atom-map references that ops can carry — drawn from the executor's
# op-dict schema (see synthelite/utils/atomic_op_executor.py).
_OP_MAP_KEYS = (
    "map_idx",
    "map_a",
    "map_b",
    "map_root_idx",
    "map_ref_a",
    "map_ref_b",
)


def _collect_op_map_ids(ops: Iterable[dict[str, Any]]) -> set[int]:
    """Pull every atom-map id referenced by any op in ``ops``."""
    maps: set[int] = set()
    for op in ops:
        for k in _OP_MAP_KEYS:
            v = op.get(k)
            if isinstance(v, int) and v > 0:
                maps.add(v)
    return maps


def _introduced_map_ids(ops: Iterable[dict[str, Any]]) -> set[int]:
    """Atom-map ids brought into existence by this step's ``add_group`` ops.

    These are the explicit maps written into a fragment SMILES (e.g. the
    ``98``/``99`` in ``*[S:98]CC[S:99]``). A later op in the *same* step may
    reference them (e.g. ``add_bond`` closing a ring onto ``99``), so they show
    up in ``_collect_op_map_ids`` — but they only come into being once the
    ``add_group`` executes, so the frontier owner check must not demand they
    pre-exist.
    """
    out: set[int] = set()
    for op in ops:
        if op.get("op") == "add_group":
            frag = op.get("fragment_smiles") or ""
            out |= {int(x) for x in _FRAG_MAP_RE.findall(frag)}
    return out


def _required_map_ids(ops: Iterable[dict[str, Any]]) -> set[int]:
    """Map ids the product mol must already carry for a step to apply.

    All referenced ids minus the ones this step introduces via ``add_group``.
    """
    ops = list(ops)
    return _collect_op_map_ids(ops) - _introduced_map_ids(ops)


def _assign_offset_maps_inplace(mol: Chem.Mol, exclude_nums: set[int]) -> None:
    """Assign atom-map numbers to any map=0 atom in ``mol``.

    Mirrors ``RetroReaction._update_unmapped_atom_num`` so that route_ops
    re-execution produces the SAME atom-map space as the original tree
    generated through ``SmilesBasedRetroReaction._apply``. The starting
    offset is ``max(existing maps in mol)+1`` (or 1), and each assigned
    id is also added to ``exclude_nums`` so sibling fragments do not
    collide on subsequent calls.
    """
    existing = {a.GetAtomMapNum() for a in mol.GetAtoms() if 0 < a.GetAtomMapNum() < 900}
    # Reserve every explicit map already on this fragment (surviving parent
    # maps *and* explicit fragment maps like :98/:99 introduced by add_group)
    # so no sibling fragment or later step is handed one of them.
    exclude_nums |= existing
    offset = max(existing) + 1 if existing else 1
    for atom in mol.GetAtoms():
        if 0 < atom.GetAtomMapNum() < 900:
            continue
        while offset in exclude_nums:
            offset += 1
        atom.SetAtomMapNum(offset)
        exclude_nums.add(offset)


def op_plan_to_tree(
    plan: dict[str, Any],
    stock_check: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Materialise a plan into a nested route-tree dict.

    Parameters
    ----------
    plan
        Plan dict as produced by ``tree_to_op_plan``.
    stock_check
        Optional callable ``(canonical_smiles) -> bool`` that reports
        whether a mol is in stock. Defaults to "everything out of stock".
        Pass ``lambda s: mol in finder.stock`` to wire in a real check.

    Returns
    -------
    dict
        A route-tree dict matching ``ReactionTree.to_dict()`` shape:
        root ``{"type": "mol", "smiles": ..., "children": [...]}`` with
        alternating reaction / mol descendants.
    """
    target_mapped = plan["target"]
    steps = plan.get("steps", [])
    in_stock_fn = stock_check or (lambda _smi: False)

    def make_mol_node(mapped_smi: str) -> dict[str, Any]:
        plain = _strip_maps_to_canonical(mapped_smi)
        return {
            "type": "mol",
            "smiles": plain,
            "is_chemical": True,
            "in_stock": in_stock_fn(plain),
            "children": [],
            # Internal: kept on the node while building so later steps can
            # resolve op map_idx → frontier mol. Stripped before return.
            "_mapped_smi": mapped_smi,
        }

    root = make_mol_node(target_mapped)
    frontier: list[dict[str, Any]] = [root]

    # Global set of map ids in use anywhere on the frontier — passed to
    # `_assign_offset_maps_inplace` so newly-introduced atoms don't collide
    # with maps already attached to siblings or other live mols.
    used_maps: set[int] = set(_maps_in(target_mapped))

    for step in steps:
        ops = step.get("ops") or []
        if any(o.get("op") == "__interpolated__" for o in ops):
            raise ValueError(
                f"step {step.get('id')}: ops contain '__interpolated__' "
                "placeholder — plan was built from a tree whose cached "
                "step lacked operations metadata; cannot replay deterministically"
            )

        req_maps = _required_map_ids(ops)

        # Find the frontier mol that owns these map ids.
        product_idx = None
        for i, mol_node in enumerate(frontier):
            if req_maps.issubset(_maps_in(mol_node["_mapped_smi"])):
                product_idx = i
                break
        if product_idx is None:
            raise ValueError(
                f"step {step.get('id')}: op map ids {sorted(req_maps)} not "
                f"found in any frontier mol (frontier size {len(frontier)})"
            )

        product = frontier.pop(product_idx)
        reactants_mapped, err = execute_ops(ops, product["_mapped_smi"])
        if err:
            raise ValueError(f"step {step.get('id')}: execute_ops failed: {err}")

        # Split reactant SMILES into fragments via RDKit (handles '.' inside
        # bracket atoms; naive str.split would mis-handle multi-fragment
        # mols with charged atoms).
        rmol = Chem.MolFromSmiles(reactants_mapped)
        if rmol is None:
            raise ValueError(
                f"step {step.get('id')}: produced unparseable reactant SMILES: {reactants_mapped!r}"
            )
        frag_mols = Chem.GetMolFrags(rmol, asMols=True, sanitizeFrags=True)
        # Assign offset map ids to any map=0 atom introduced by add_group
        # (mirrors `RetroReaction._update_unmapped_atom_num`). Per-fragment,
        # but with a shared `used_maps` so siblings don't collide.
        for fm in frag_mols:
            _assign_offset_maps_inplace(fm, used_maps)
        reactant_nodes = [make_mol_node(Chem.MolToSmiles(fm)) for fm in frag_mols]

        # Build the reaction node connecting product to its reactants.
        product_plain = product["smiles"]
        reactants_plain_join = ".".join(c["smiles"] for c in reactant_nodes)
        # Use the offset-mapped fragment SMILES (not the pre-offset
        # `reactants_mapped`) so the stored `mapped_reaction_smiles`
        # matches what later route_ops passes will see.
        reactants_mapped_offset = ".".join(c["_mapped_smi"] for c in reactant_nodes)
        rxn_metadata: dict[str, Any] = {
            "operations": ops,
            "mapped_reaction_smiles": f"{product['_mapped_smi']}>>{reactants_mapped_offset}",
            "reaction_prompt": step.get("desc", ""),
            "from_route_ops_plan": True,
        }
        if step.get("id"):
            rxn_metadata["step_id"] = step["id"]
        if step.get("step_rationale"):
            rxn_metadata["step_rationale"] = step["step_rationale"]
        if step.get("reaction_conditions"):
            rxn_metadata["reaction_conditions"] = step["reaction_conditions"]
        if step.get("kind"):
            rxn_metadata["kind"] = step["kind"]
        rxn_node: dict[str, Any] = {
            "type": "reaction",
            "smiles": f"{product_plain}>>{reactants_plain_join}",
            "is_reaction": True,
            "metadata": rxn_metadata,
            "children": reactant_nodes,
        }
        product["children"] = [rxn_node]
        frontier.extend(reactant_nodes)

    # Strip the internal _mapped_smi marker before returning so the
    # output exactly matches ReactionTree.to_dict() shape.
    def _strip_internal(node: dict[str, Any]) -> None:
        node.pop("_mapped_smi", None)
        for c in node.get("children") or []:
            _strip_internal(c)

    _strip_internal(root)
    return root
