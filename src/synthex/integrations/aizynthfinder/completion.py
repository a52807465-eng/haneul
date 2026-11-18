"""Ordinary AiZynthFinder searches used to finish strategic route leaves."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

from aizynthfinder.aizynthfinder import AiZynthFinder
from aizynthfinder.chem import Molecule

from synthex.completion import CompletionBudget, LeafSearchResult

AIZYNTHFINDER_VERSION = "4.4.1"


class _SelectableCollection(Protocol):
    @property
    def items(self) -> list[str]: ...

    def select(self, value: list[str]) -> None: ...


def _select_loaded(
    collection: _SelectableCollection,
    requested: list[str] | None,
    *,
    label: str,
    required: bool,
) -> None:
    selected = list(requested) if requested is not None else list(collection.items)
    if required and not selected:
        raise ValueError(f"AiZynthFinder configuration loads no {label}")
    if selected:
        try:
            collection.select(selected)
        except KeyError as exc:
            raise ValueError(
                f"unknown AiZynthFinder {label}; requested {selected}, available {collection.items}"
            ) from exc


class AiZynthFinderLeafCompleter:
    """Run the pinned upstream planner once for each requested leaf.

    The upstream object and its loaded policies/stock are reused, while setting a
    new target starts a fresh search tree and ``prepare_tree`` clears policy caches.
    """

    def __init__(
        self,
        config_path: str | Path,
        budget: CompletionBudget | None = None,
        *,
        stocks: list[str] | None = None,
        expansion_policies: list[str] | None = None,
        filter_policies: list[str] | None = None,
    ) -> None:
        installed = version("aizynthfinder")
        if installed != AIZYNTHFINDER_VERSION:
            raise RuntimeError(
                f"SynthEx requires aizynthfinder=={AIZYNTHFINDER_VERSION}; found {installed}"
            )
        self.budget = budget or CompletionBudget()
        self._finder = AiZynthFinder(configfile=str(config_path))
        # AiZynthFinder deliberately separates loading from selection. Select all
        # loaded entries by default, including a ring-breaker policy when present.
        _select_loaded(self._finder.stock, stocks, label="stocks", required=True)
        _select_loaded(
            self._finder.expansion_policy,
            expansion_policies,
            label="expansion policies",
            required=True,
        )
        _select_loaded(
            self._finder.filter_policy,
            filter_policies,
            label="filter policies",
            required=True,
        )
        search = self._finder.config.search
        search.max_transforms = self.budget.max_transforms
        search.iteration_limit = self.budget.iteration_limit
        search.time_limit = self.budget.time_limit
        search.return_first = self.budget.return_first

    def in_stock(self, smiles: str) -> bool:
        """Check a molecule against the same stock used for completion."""

        return Molecule(smiles=smiles) in self._finder.stock

    def search(self, smiles: str) -> LeafSearchResult:
        """Run upstream MCTS and return the first solved route fragment."""

        canonical = Molecule(smiles=smiles).smiles
        try:
            self._finder.target_smiles = canonical
            self._finder.tree_search(show_progress=False)
            self._finder.build_routes()
            solved_tree = next(
                (tree for tree in self._finder.routes.reaction_trees if tree.is_solved),
                None,
            )
            statistics: dict[str, Any] = self._finder.extract_statistics()
            if solved_tree is None:
                return LeafSearchResult(
                    smiles=canonical,
                    status="unsolved",
                    statistics=statistics,
                    backend="aizynthfinder",
                    backend_version=AIZYNTHFINDER_VERSION,
                )
            return LeafSearchResult(
                smiles=canonical,
                status="solved",
                route_tree=solved_tree.to_dict(include_metadata=True),
                statistics=statistics,
                backend="aizynthfinder",
                backend_version=AIZYNTHFINDER_VERSION,
            )
        except Exception as exc:
            return LeafSearchResult(
                smiles=canonical,
                status="failed",
                backend="aizynthfinder",
                backend_version=AIZYNTHFINDER_VERSION,
                error=f"{type(exc).__name__}: {exc}",
            )


__all__ = ["AIZYNTHFINDER_VERSION", "AiZynthFinderLeafCompleter", "CompletionBudget"]
