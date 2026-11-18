from __future__ import annotations

import inspect
from importlib.metadata import version
from types import SimpleNamespace

import pytest
from aizynthfinder.aizynthfinder import AiZynthFinder
from aizynthfinder.context.config import Configuration

from synthex.integrations.aizynthfinder import AIZYNTHFINDER_VERSION, CompletionBudget
from synthex.integrations.aizynthfinder import completion as adapter


class _FakeCollection:
    def __init__(self, items: list[str]) -> None:
        self.items = items
        self.selection: list[str] = []

    def select(self, value: list[str]) -> None:
        unknown = set(value) - set(self.items)
        if unknown:
            raise KeyError(next(iter(unknown)))
        self.selection = list(value)


class _FakeFinder:
    def __init__(self, configfile: str) -> None:
        self.configfile = configfile
        self.stock = _FakeCollection(["zinc", "emolecules"])
        self.expansion_policy = _FakeCollection(["uspto", "ringbreaker"])
        self.filter_policy = _FakeCollection(["uspto"])
        self.config = SimpleNamespace(
            search=SimpleNamespace(
                max_transforms=0,
                iteration_limit=0,
                time_limit=0,
                return_first=False,
            )
        )


def test_exact_upstream_version_is_installed():
    assert version("aizynthfinder") == AIZYNTHFINDER_VERSION == "4.4.1"


def test_completion_budget_matches_manuscript():
    budget = CompletionBudget()
    assert (budget.max_transforms, budget.iteration_limit, budget.time_limit) == (6, 500, 1200)
    assert budget.return_first


def test_upstream_seams_used_by_completion_adapter_are_present():
    assert "show_progress" in inspect.signature(AiZynthFinder.tree_search).parameters
    assert hasattr(AiZynthFinder, "build_routes")
    config = Configuration()
    assert hasattr(config.search, "max_transforms")
    assert hasattr(config.search, "iteration_limit")
    assert hasattr(config.search, "time_limit")
    assert hasattr(config.search, "return_first")


def test_adapter_selects_all_loaded_entries_and_applies_budget(monkeypatch):
    monkeypatch.setattr(adapter, "AiZynthFinder", _FakeFinder)
    monkeypatch.setattr(adapter, "version", lambda _distribution: "4.4.1")

    completer = adapter.AiZynthFinderLeafCompleter("aizynthfinder.yml")

    assert completer._finder.stock.selection == ["zinc", "emolecules"]
    assert completer._finder.expansion_policy.selection == ["uspto", "ringbreaker"]
    assert completer._finder.filter_policy.selection == ["uspto"]
    assert completer._finder.config.search.max_transforms == 6
    assert completer._finder.config.search.iteration_limit == 500
    assert completer._finder.config.search.time_limit == 1200
    assert completer._finder.config.search.return_first


def test_adapter_rejects_unknown_selection(monkeypatch):
    monkeypatch.setattr(adapter, "AiZynthFinder", _FakeFinder)
    monkeypatch.setattr(adapter, "version", lambda _distribution: "4.4.1")

    with pytest.raises(ValueError, match="unknown AiZynthFinder stocks"):
        adapter.AiZynthFinderLeafCompleter("aizynthfinder.yml", stocks=["missing"])
