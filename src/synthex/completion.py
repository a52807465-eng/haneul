"""Dependency-neutral leaf completion and route stitching."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field
from rdkit import Chem

from synthex.routejson import Route, op_plan_to_tree


class LeafSearchResult(BaseModel):
    """Result of one template search for one unique strategic leaf."""

    model_config = ConfigDict(extra="forbid")

    smiles: str
    status: Literal["solved", "unsolved", "failed"]
    route_tree: dict[str, Any] | None = None
    statistics: dict[str, Any] = Field(default_factory=dict)
    backend: str
    backend_version: str
    error: str | None = None


class CompletionBudget(BaseModel):
    """Per-leaf limits reported in Methods section 4.12."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_transforms: int = Field(default=6, ge=1)
    iteration_limit: int = Field(default=500, ge=1)
    time_limit: int = Field(default=1200, ge=1)
    return_first: bool = True


class LeafCompleter(Protocol):
    """Small boundary implemented by the AiZynthFinder adapter."""

    def in_stock(self, smiles: str) -> bool: ...

    def search(self, smiles: str) -> LeafSearchResult: ...


class CompletionRecord(BaseModel):
    """Provenance for grafting one unique leaf zero or more times."""

    model_config = ConfigDict(extra="forbid")

    leaf_smiles: str
    occurrences: int = Field(ge=1)
    result: LeafSearchResult


class StitchedRoute(BaseModel):
    """Strategic ReactionJSON route plus optional AiZynthFinder fragments."""

    model_config = ConfigDict(extra="forbid")

    format_version: Literal["stitched-route-1.0"] = "stitched-route-1.0"
    strategic_route: Route
    strategic_tree: dict[str, Any]
    stitched_tree: dict[str, Any]
    completions: list[CompletionRecord] = Field(default_factory=list)
    is_solved: bool


def canonical_smiles(smiles: str) -> str:
    """Return a canonical, atom-map-free SMILES string."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Could not parse leaf SMILES: {smiles!r}")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(molecule)


def _molecule_leafs(tree: dict[str, Any]) -> list[dict[str, Any]]:
    leaves: list[dict[str, Any]] = []

    def visit(node: dict[str, Any]) -> None:
        children = node.get("children") or []
        if node.get("type") == "mol" and not children:
            leaves.append(node)
            return
        for child in children:
            visit(child)

    visit(tree)
    return leaves


def _graft(fragment: dict[str, Any], destination: dict[str, Any], record: CompletionRecord) -> None:
    if fragment.get("type") != "mol":
        raise ValueError("AiZynthFinder fragment must be rooted at a molecule node")
    if canonical_smiles(fragment.get("smiles", "")) != record.leaf_smiles:
        raise ValueError("AiZynthFinder fragment root does not match the leaf it is completing")
    children = deepcopy(fragment.get("children") or [])
    if not children:
        raise ValueError("A solved completion fragment must contain at least one reaction")
    for child in children:
        metadata = child.setdefault("metadata", {})
        metadata["synthex_completion"] = {
            "backend": record.result.backend,
            "backend_version": record.result.backend_version,
            "leaf_smiles": record.leaf_smiles,
        }
    destination["children"] = children


def _tree_is_solved(tree: dict[str, Any], completer: LeafCompleter) -> bool:
    leaves = _molecule_leafs(tree)
    if not leaves:
        return False
    solved = True
    for leaf in leaves:
        available = completer.in_stock(canonical_smiles(leaf.get("smiles", "")))
        leaf["in_stock"] = available
        solved = solved and available
    return solved


def complete_route(route: Route | dict[str, Any], completer: LeafCompleter) -> StitchedRoute:
    """Complete every unique non-stock strategic leaf and graft solved fragments.

    Exactly one template search is performed for each unique canonical leaf. A
    successful fragment is copied to every occurrence of that leaf. Failures are
    recorded and leave the original strategic route intact.
    """

    parsed = route if isinstance(route, Route) else Route.model_validate(route)
    strategic_tree = op_plan_to_tree(parsed.to_op_plan(), stock_check=completer.in_stock)
    stitched_tree = deepcopy(strategic_tree)

    occurrences: dict[str, list[dict[str, Any]]] = {}
    for leaf in _molecule_leafs(stitched_tree):
        leaf_smiles = canonical_smiles(leaf.get("smiles", ""))
        if not completer.in_stock(leaf_smiles):
            occurrences.setdefault(leaf_smiles, []).append(leaf)

    records: list[CompletionRecord] = []
    for leaf_smiles, destinations in occurrences.items():
        try:
            result = completer.search(leaf_smiles)
        except Exception as exc:  # adapters must not erase a usable strategic route
            result = LeafSearchResult(
                smiles=leaf_smiles,
                status="failed",
                backend=type(completer).__name__,
                backend_version="unknown",
                error=f"{type(exc).__name__}: {exc}",
            )
        record = CompletionRecord(
            leaf_smiles=leaf_smiles,
            occurrences=len(destinations),
            result=result,
        )
        records.append(record)
        if result.status == "solved" and result.route_tree is not None:
            for destination in destinations:
                _graft(result.route_tree, destination, record)

    return StitchedRoute(
        strategic_route=parsed,
        strategic_tree=strategic_tree,
        stitched_tree=stitched_tree,
        completions=records,
        is_solved=_tree_is_solved(stitched_tree, completer),
    )


__all__ = [
    "CompletionRecord",
    "CompletionBudget",
    "LeafCompleter",
    "LeafSearchResult",
    "StitchedRoute",
    "canonical_smiles",
    "complete_route",
]
