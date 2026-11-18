"""Rewrite a route's op atom-map references into the renderer's numbering.

The code-ops search policy (``EditBasedRetroReaction._apply``) numbers atoms
introduced by ``add_group`` with a per-reaction counter that *reuses* map ids
freed by earlier removals. ``render_plan`` / ``op_plan_to_tree``, by contrast,
thread one global monotonic counter across the whole route and never reuse a
freed id. So the raw ops copied out of a search tree reference numbers that the
renderer never assigns — a later op points at an atom that has a different id
under the renderer's scheme, and the plan fails to render.

This module reconciles the two: it replays the route through the renderer's
exact numbering and rewrites every op's map-reference *key* into the renderer's
id space. The result is a plan whose ops resolve deterministically under
``render_plan`` and whose atom maps are globally unique across the whole route
(so the editing agent can reorder / insert steps without id clashes).

The search reuses freed ids *across coexisting molecules*: after a convergent
split (Suzuki, amide coupling, …) branch A and branch B both live on the
frontier, and a step on branch B may reuse a search id that branch A still
carries for a different atom. So a search id does not identify a molecule — the
same id can name different atoms on two live molecules at once, and routing a
step by its ids picks the wrong branch. We therefore never route by ids. Each
frontier entry carries the molecule in *both* numberings; a step's product is
located by matching its ``mapped_reaction_smiles`` product against the frontier
by map-stripped canonical structure, that one molecule's search→global map is
derived from its own two representations, its ops are rewritten through that
local map, and the new reactants are paired search-to-global (again by
structure) into fresh entries. Deriving the map per molecule from structure —
never from ids — keeps each branch's numbering isolated.

Explicit fragment maps (e.g. the ``:98``/``:99`` in ``*[S:98]CC[S:99]``) are
left untouched: the renderer preserves them verbatim, and
``_assign_offset_maps_inplace`` now reserves them so nothing else reuses them.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from rdkit import Chem

from synthex.reactionjson.executor import execute_ops
from synthex.routejson.executor import (
    _OP_MAP_KEYS,
    _assign_offset_maps_inplace,
    _maps_in,
)


def _rewrite_op_refs(ops: list[dict[str, Any]], s2g: dict[int, int]) -> list[dict[str, Any]]:
    """Return a copy of ``ops`` with every map-reference key mapped through s2g.

    Only the reference keys (``map_a``, ``map_idx``, …) are rewritten; the
    ``fragment_smiles`` explicit maps are left as-is (the renderer preserves
    them). A reference missing from ``s2g`` is left unchanged so the failure
    surfaces later as a normal render error rather than a silent mis-map.
    """
    out = deepcopy(ops)
    for op in out:
        for k in _OP_MAP_KEYS:
            v = op.get(k)
            if isinstance(v, int) and v in s2g:
                op[k] = s2g[v]
    return out


def _strip(mol: Chem.Mol) -> Chem.Mol:
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    return m


def _canonical(smi: str) -> str | None:
    """Map-stripped canonical SMILES, or ``None`` if unparseable."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(_strip(mol))


def _local_map(search_mol: Chem.Mol, global_mol: Chem.Mol) -> dict[int, int] | None:
    """search-map -> global-map for one molecule given in both numberings.

    The two mols are the same structure numbered differently; a substructure
    match (maps ignored) pairs their atoms. Returns ``None`` if they don't
    match — the caller then treats the route as un-propagatable.
    """
    match = global_mol.GetSubstructMatch(_strip(search_mol))
    if not match:
        return None
    local: dict[int, int] = {}
    for i, j in enumerate(match):
        sm = search_mol.GetAtomWithIdx(i).GetAtomMapNum()
        gm = global_mol.GetAtomWithIdx(j).GetAtomMapNum()
        if sm > 0 and gm > 0:
            local[sm] = gm
    return local


def _pair_reactants(search_smi: str, global_frags: list[Chem.Mol]) -> list[tuple[str, str]] | None:
    """Pair search reactant fragments with their global counterparts.

    Returns one ``(search_mapped_smiles, global_mapped_smiles)`` per search
    fragment, matched by map-stripped canonical structure, or ``None`` if any
    fragment has no structural match among ``global_frags`` (which means the
    rewritten ops produced different chemistry — bail rather than corrupt the
    frontier).
    """
    s_frags = list(
        Chem.GetMolFrags(Chem.MolFromSmiles(search_smi), asMols=True, sanitizeFrags=True)
    )
    g_by_key: dict[str, list[Chem.Mol]] = {}
    for gf in global_frags:
        g_by_key.setdefault(Chem.MolToSmiles(_strip(gf)), []).append(gf)

    out: list[tuple[str, str]] = []
    for sf in s_frags:
        bucket = g_by_key.get(Chem.MolToSmiles(_strip(sf)))
        if not bucket:
            return None
        gf = bucket.pop(0)
        out.append((Chem.MolToSmiles(sf), Chem.MolToSmiles(gf)))
    return out


def propagate_map_ids(
    target: str, steps: list[dict[str, Any]]
) -> list[list[dict[str, Any]]] | None:
    """Rewrite each step's ops into the renderer's atom-map numbering.

    Parameters
    ----------
    target
        Mapped SMILES of the target (search numbering). Its ids are the seed of
        the global space (identity), matching how the renderer starts.
    steps
        One dict per step with keys ``ops`` (search-numbered) and
        ``mapped_reaction_smiles`` (``product>>reactants``, search numbering).

    Returns
    -------
    list per step of rewritten ops, or ``None`` if the route can't be
    propagated (missing reaction SMILES, ``__interpolated__`` placeholder, an op
    that fails to execute, or a product/reactant that can't be matched by
    structure) — the caller should then keep the ops verbatim.
    """
    used: set[int] = set(_maps_in(target))
    # Each frontier entry carries the molecule in BOTH numberings:
    # (search_mapped_smiles, global_mapped_smiles). The target seeds the global
    # space as identity, so its two forms are the same string.
    frontier: list[tuple[str, str]] = [(target, target)]
    rewritten: list[list[dict[str, Any]]] = []

    for step in steps:
        ops = step.get("ops") or []
        mrs = step.get("mapped_reaction_smiles") or ""
        if ">>" not in mrs or any(o.get("op") == "__interpolated__" for o in ops):
            return None
        product_search, reactants_search = mrs.split(">>", 1)

        # Locate the product by STRUCTURE (map ids can't disambiguate — the
        # search reuses them across coexisting molecules). The tree gives the
        # product SMILES directly, so match it against the frontier.
        pkey = _canonical(product_search)
        if pkey is None:
            return None
        pidx = next(
            (i for i, (s_smi, _g) in enumerate(frontier) if _canonical(s_smi) == pkey),
            None,
        )
        if pidx is None:
            return None
        s_smi, g_smi = frontier.pop(pidx)

        # Translate this one molecule's search ids into the global space from its
        # own two representations — a reused id on another branch can't leak in.
        local = _local_map(Chem.MolFromSmiles(s_smi), Chem.MolFromSmiles(g_smi))
        if local is None:
            return None
        ops_global = _rewrite_op_refs(ops, local)
        rewritten.append(ops_global)

        reactants_global, err = execute_ops(ops_global, g_smi)
        if err:
            return None
        rmol = Chem.MolFromSmiles(reactants_global)
        if rmol is None:
            return None
        frags = Chem.GetMolFrags(rmol, asMols=True, sanitizeFrags=True)
        for fm in frags:
            _assign_offset_maps_inplace(fm, used)

        paired = _pair_reactants(reactants_search, list(frags))
        if paired is None:
            return None
        frontier.extend(paired)

    return rewritten
