"""Fast tree.json → route-tree extraction without loading the full Configuration.

``LLMGuidedBeamSearchTree.from_json`` is the canonical loader but it pulls in
the whole search stack: template-search engine (42k templates × 1536D
embeddings), 22M-compound stock CSV, ONNX filter model, LLM client init.
That's all wasted work when we only want to extract route_dicts for the
route_ops pipeline.

This module reads the JSON directly. Each leaf of the search tree yields one
route_dict (same shape as ``ReactionTree.to_dict()``) so the existing
``tree_to_op_plan`` pipeline consumes it unchanged.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rdkit import Chem

_MAP_RE = re.compile(r":(\d+)\]")


def _strip_maps_to_canonical(maybe_mapped: str) -> str:
    mol = Chem.MolFromSmiles(maybe_mapped)
    if mol is None:
        return maybe_mapped
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def load_route_trees_from_tree_json(
    path: str | Path | None = None,
    data: dict[str, Any] | None = None,
    stock_check: Callable[[str], bool] | None = None,
) -> list[dict[str, Any]]:
    """Parse a serialised search tree and return one route_dict per leaf.

    Parameters
    ----------
    path
        Path to a ``tree.{n}.llm_guided.json`` file produced by
        any ``SearchTree.serialize`` (beam search or MCTS — same on-disk format).
    stock_check
        Optional ``(canonical_smiles) -> bool``. Used to set ``in_stock``
        on each mol node. Defaults to "everything out of stock".

    Returns
    -------
    list[dict]
        One route_dict per leaf (a node with no expanded children).
        Each dict matches ``ReactionTree.to_dict()`` shape.
    """
    if data is not None and path is not None:
        raise ValueError("Provide exactly one of path or data, not both")
    if data is None:
        with open(path) as f:
            data = json.load(f)

    # Molecule store: JSON keys are strings even when the original was an int
    # (id() returns an int but json.dumps converts dict keys to str).
    mol_store: dict[int, dict[str, Any]] = {int(k): v for k, v in data["molecules"].items()}

    in_stock_fn = stock_check or (lambda _smi: False)

    def make_mol_node(mol_id: int) -> dict[str, Any]:
        spec = mol_store[mol_id]
        # Per ``MoleculeSerializer._add_mol``: root mol carries original
        # SMILES, child mols carry the mapped form.
        mapped_or_plain = spec["smiles"]
        plain = _strip_maps_to_canonical(mapped_or_plain)
        return {
            "type": "mol",
            "smiles": plain,
            "is_chemical": True,
            "in_stock": in_stock_fn(plain),
            "children": [],
            # internal: used while building the tree, stripped before return.
            "_mapped_smi": mapped_or_plain,
            "_mol_id": mol_id,
        }

    def build_route_from_action_path(
        root_mol_id: int, action_path: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Reconstruct a route_dict from a flat list of actions taken from
        root down to a leaf."""
        root_mol = make_mol_node(root_mol_id)
        mol_by_id: dict[int, dict[str, Any]] = {root_mol_id: root_mol}

        for action in action_path:
            parent_id = action["mol"]
            parent_node = mol_by_id.get(parent_id)
            if parent_node is None:
                # Defensive: should never happen if the MCTS tree is well-formed.
                raise ValueError(f"action references mol_id {parent_id} not in current frontier")
            outcome_idx = action.get("index", 0)
            reactant_ids_outcomes = action.get("reactants") or []
            if not reactant_ids_outcomes:
                # Unqueried action / leaf — skip.
                continue
            reactant_ids = reactant_ids_outcomes[outcome_idx]
            reactant_nodes = [make_mol_node(rid) for rid in reactant_ids]

            reactants_plain = ".".join(r["smiles"] for r in reactant_nodes)
            metadata = dict(action.get("metadata") or {})
            # ReactionTree.to_dict() populates ``mapped_reaction_smiles`` from
            # the product's + reactants' mapped_smiles. The on-disk action
            # serialisation only carries ``mapped_product_smiles``, so we
            # reconstruct the full mapped reaction here.
            if "mapped_reaction_smiles" not in metadata:
                product_mapped = (
                    metadata.get("mapped_product_smiles") or parent_node.get("_mapped_smi") or ""
                )
                reactants_mapped = ".".join(r.get("_mapped_smi", "") for r in reactant_nodes)
                if product_mapped and reactants_mapped:
                    metadata["mapped_reaction_smiles"] = f"{product_mapped}>>{reactants_mapped}"
            rxn_node = {
                "type": "reaction",
                "smiles": f"{parent_node['smiles']}>>{reactants_plain}",
                "is_reaction": True,
                "metadata": metadata,
                "children": reactant_nodes,
            }
            parent_node["children"] = [rxn_node]

            # Update frontier mapping: the consumed parent is gone, its reactants take its slot.
            del mol_by_id[parent_id]
            for r_id, r_node in zip(reactant_ids, reactant_nodes, strict=True):
                mol_by_id[r_id] = r_node

        # Strip internal markers before returning.
        def _strip_internal(node: dict[str, Any]) -> None:
            node.pop("_mapped_smi", None)
            node.pop("_mol_id", None)
            for c in node.get("children") or []:
                _strip_internal(c)

        _strip_internal(root_mol)
        return root_mol

    # Walk the search tree depth-first, collecting one action-path per leaf.
    root_search_node = data["tree"]
    root_mol_id = root_search_node["state"]["mols"][0]

    routes: list[dict[str, Any]] = []

    def walk(node: dict[str, Any], action_path: list[dict[str, Any]]) -> None:
        children = node.get("children") or []
        actions = node.get("children_actions") or []
        expanded = [i for i, c in enumerate(children) if c is not None]
        if not expanded:
            routes.append(build_route_from_action_path(root_mol_id, action_path))
            return
        for i in expanded:
            walk(children[i], action_path + [actions[i]])

    walk(root_search_node, [])
    return routes


def load_solved_route_trees_from_tree_json(
    path: str | Path,
    stock_check: Callable[[str], bool],
) -> list[dict[str, Any]]:
    """Like ``load_route_trees_from_tree_json`` but returns only routes whose
    leaves are entirely in-stock. ``stock_check`` is mandatory because
    "solved" is defined relative to the stock."""
    routes = load_route_trees_from_tree_json(path, stock_check=stock_check)
    return [r for r in routes if _is_solved_route(r)]


def _is_solved_route(route: dict[str, Any]) -> bool:
    """True if every leaf mol node has ``in_stock=True``."""
    if route.get("type") != "mol":
        return False
    children = route.get("children") or []
    if not children:
        return bool(route.get("in_stock"))
    # If this mol has been expanded, recurse into the reaction's reactant mols.
    rxn = children[0]
    return all(_is_solved_route(child) for child in (rxn.get("children") or []))
