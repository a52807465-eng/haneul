from __future__ import annotations

from synthex.completion import LeafSearchResult, complete_route
from synthex.routejson import Route


class FakeCompleter:
    def __init__(self, fail: bool = False):
        self.calls: list[str] = []
        self.fail = fail

    def in_stock(self, smiles: str) -> bool:
        return smiles == "C"

    def search(self, smiles: str) -> LeafSearchResult:
        self.calls.append(smiles)
        if self.fail:
            raise RuntimeError("fixture failure")
        return LeafSearchResult(
            smiles=smiles,
            status="solved",
            backend="fake",
            backend_version="1",
            route_tree={
                "type": "mol",
                "smiles": smiles,
                "is_chemical": True,
                "in_stock": False,
                "children": [
                    {
                        "type": "reaction",
                        "smiles": f"{smiles}>>C.C",
                        "is_reaction": True,
                        "metadata": {},
                        "children": [
                            {"type": "mol", "smiles": "C", "is_chemical": True, "in_stock": True},
                            {"type": "mol", "smiles": "C", "is_chemical": True, "in_stock": True},
                        ],
                    }
                ],
            },
        )


def _route_with_two_equal_unsolved_leaves() -> Route:
    return Route.model_validate(
        {
            "target": "[CH3:1][CH2:2][CH2:3][CH3:4]",
            "steps": [
                {
                    "id": "S1",
                    "operations": [{"op": "break_bond", "map_a": 2, "map_b": 3}],
                }
            ],
        }
    )


def test_completion_searches_each_unique_leaf_once_and_grafts_every_occurrence():
    completer = FakeCompleter()
    result = complete_route(_route_with_two_equal_unsolved_leaves(), completer)

    assert completer.calls == ["CC"]
    assert result.completions[0].occurrences == 2
    assert result.is_solved
    grafts = result.stitched_tree["children"][0]["children"]
    assert all(node["children"] for node in grafts)
    assert result.strategic_tree != result.stitched_tree


def test_failed_completion_preserves_strategic_route():
    completer = FakeCompleter(fail=True)
    result = complete_route(_route_with_two_equal_unsolved_leaves(), completer)

    assert not result.is_solved
    assert result.stitched_tree == result.strategic_tree
    assert result.completions[0].result.status == "failed"
    assert "fixture failure" in (result.completions[0].result.error or "")
