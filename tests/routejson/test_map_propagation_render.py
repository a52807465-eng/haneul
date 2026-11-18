"""Regression tests: a search-numbered route must render after conversion.

The code-ops search policy numbers atoms with a per-reaction counter that
*reuses* atom-map ids freed by an earlier removal; the renderer uses one global
monotonic counter and never reuses a freed id. Copying the search ops verbatim
into a plan therefore left later ops pointing at ids the renderer never
assigns, and ``render_plan`` raised "op map ids [...] not in any frontier mol".

``tree_to_op_plan`` now rewrites the op references into the renderer's numbering
(``propagate_map_ids``). These tests build synthetic search-numbered trees that
reproduce the two failure modes seen in production and assert the converted
plan renders with the original chemistry intact.
"""

from __future__ import annotations

import unittest

from rdkit import Chem, RDLogger

from synthex.routejson import tree_to_op_plan
from synthex.routejson.render import render_plan

RDLogger.DisableLog("rdApp.*")


def _strip(mapped_smiles: str) -> str:
    mol = Chem.MolFromSmiles(mapped_smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def _mol(smiles: str, in_stock: bool = False, children=None) -> dict:
    return {
        "type": "mol",
        "smiles": smiles,
        "is_chemical": True,
        "in_stock": in_stock,
        "children": children or [],
    }


def _rxn(ops: list[dict], mapped_reaction_smiles: str, children=None) -> dict:
    product, reactants = mapped_reaction_smiles.split(">>", 1)
    smiles = f"{_strip(product)}>>{_strip(reactants)}"
    return {
        "type": "reaction",
        "smiles": smiles,
        "is_reaction": True,
        "metadata": {
            "operations": ops,
            "mapped_reaction_smiles": mapped_reaction_smiles,
            "reaction_prompt": "synthetic step",
        },
        "children": children or [],
    }


def _linear_tree(steps: list[tuple[list[dict], str]]) -> dict:
    """Build a linear route tree from ``(ops, mapped_reaction_smiles)`` steps.

    ``steps`` run product -> reactants top-down; the leaf is marked in_stock.
    """
    last_reactants = steps[-1][1].split(">>", 1)[1]
    node = _mol(_strip(last_reactants), in_stock=True)
    for ops, mrs in reversed(steps):
        product = mrs.split(">>", 1)[0]
        node = _mol(_strip(product), children=[_rxn(ops, mrs, children=[node])])
    return node


def _reaction_keys(reaction_smiles: str) -> tuple:
    product, reactants = reaction_smiles.split(">>")

    def canon(smi: str) -> str:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol else smi

    return (canon(product), tuple(sorted(canon(r) for r in reactants.split("."))))


class TestSearchNumberingRenders(unittest.TestCase):
    def test_reuse_of_freed_map_id(self):
        """S3 reuses map 8 (freed by S2); S4 references it — must still render."""
        # Search numbering: S1 acetylates the phenol O (adds C:9/CH3:8/O:10);
        # S2 strips the acetate (frees 8,9,10); S3 methylates the phenol and the
        # new carbon *reuses* map 8; S4 edits that reused atom.
        steps = [
            (
                [{"op": "add_group", "map_idx": 7, "fragment_smiles": "*C(C)=O"}],
                "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[OH:7]"
                ">>[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[O:7][C:9]([CH3:8])=[O:10]",
            ),
            (
                [{"op": "remove_group", "map_idx": 7, "map_root_idx": 9}],
                "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[O:7][C:9]([CH3:8])=[O:10]"
                ">>[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[OH:7]",
            ),
            (
                [{"op": "add_group", "map_idx": 7, "fragment_smiles": "*C"}],
                "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[OH:7]"
                ">>[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[O:7][CH3:8]",
            ),
            (
                [{"op": "change_atom", "map_idx": 8, "symbol": "N"}],
                "[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[O:7][CH3:8]"
                ">>[cH:1]1[cH:2][cH:3][cH:4][cH:5][c:6]1[O:7][NH2:8]",
            ),
        ]
        tree = _linear_tree(steps)
        plan = tree_to_op_plan(tree)

        # Propagation rewrote S4's reference off the reused id 8.
        self.assertNotEqual(plan["steps"][3]["ops"][0]["map_idx"], 8)

        rendered = render_plan(plan, strip_frontier_maps=True)
        route = rendered["route"]
        self.assertEqual(len(route), 4)

        # Chemistry is preserved step-for-step.
        expected = [
            _reaction_keys(f"{_strip(mrs.split('>>')[0])}>>{_strip(mrs.split('>>')[1])}")
            for _, mrs in steps
        ]
        got = [_reaction_keys(entry["smiles"]) for entry in route]
        self.assertEqual(got, expected)

    def test_reuse_across_convergent_branches(self):
        """A convergent split reuses one branch's id on the *other* branch.

        After S1 splits the biaryl, branch A (phenyl bromide) and branch B
        (aryl boronic acid) both sit on the frontier. S2 edits branch A and its
        reactants reuse ids 16/17 that branch B still carries for its boron.
        S3 then edits branch B via id 16. A single flat search->global map would
        have let S2's reconcile clobber id 16, breaking S3; routing each step to
        its product by structure keeps every branch's numbering isolated.
        """
        target = (
            "[cH:1]1[cH:2][cH:3][cH:4][c:5]([OH:6])[c:7]1"
            "-[c:8]1[cH:9][cH:10][cH:11][c:12]([CH3:14])[cH:13]1"
        )
        s1 = (
            [
                {"op": "break_bond", "map_a": 7, "map_b": 8},
                {"op": "add_group", "map_idx": 7, "fragment_smiles": "*Br"},
                {"op": "add_group", "map_idx": 8, "fragment_smiles": "*B(O)O"},
            ],
            target + ">>"
            "[cH:1]1[cH:2][cH:3][cH:4][c:5]([OH:6])[c:7]1[Br:15]."
            "[c:8]1([B:16]([OH:17])[OH:18])[cH:9][cH:10][cH:11][c:12]([CH3:14])[cH:13]1",
        )
        # Branch A: swap Br for a nitro, reusing ids 15/16/17 on this molecule.
        sA = (
            [
                {"op": "remove_group", "map_idx": 7, "map_root_idx": 15},
                {"op": "add_group", "map_idx": 7, "fragment_smiles": "*[N+](=O)[O-]"},
            ],
            "[cH:1]1[cH:2][cH:3][cH:4][c:5]([OH:6])[c:7]1[Br:15]>>"
            "[cH:1]1[cH:2][cH:3][cH:4][c:5]([OH:6])[c:7]1[N+:15](=[O:16])[O-:17]",
        )
        # Branch B: swap boronic acid for Br, referencing id 16 (the boron).
        sB = (
            [
                {"op": "remove_group", "map_idx": 8, "map_root_idx": 16},
                {"op": "add_group", "map_idx": 8, "fragment_smiles": "*Br"},
            ],
            "[c:8]1([B:16]([OH:17])[OH:18])[cH:9][cH:10][cH:11][c:12]([CH3:14])[cH:13]1"
            ">>[c:8]1([Br:16])[cH:9][cH:10][cH:11][c:12]([CH3:14])[cH:13]1",
        )

        def _branch(step):
            ops, mrs = step
            leaf = _mol(_strip(mrs.split(">>", 1)[1]), in_stock=True)
            return _mol(
                _strip(mrs.split(">>", 1)[0]),
                children=[_rxn(ops, mrs, children=[leaf])],
            )

        tree = _mol(
            _strip(target),
            children=[_rxn(s1[0], s1[1], children=[_branch(sA), _branch(sB)])],
        )
        plan = tree_to_op_plan(tree)

        # Propagation resolved branch B's reused reference off id 16 instead of
        # falling back to verbatim ops (which would fail to render).
        branch_b_step = plan["steps"][2]
        self.assertEqual(branch_b_step["ops"][0]["op"], "remove_group")
        self.assertNotEqual(branch_b_step["ops"][0]["map_root_idx"], 16)

        rendered = render_plan(plan, strip_frontier_maps=True)
        got = {_reaction_keys(e["smiles"]) for e in rendered["route"]}
        expected = {
            _reaction_keys("Cc1cccc(-c2ccccc2O)c1>>Oc1ccccc1Br.Cc1cccc(B(O)O)c1"),
            _reaction_keys("Oc1ccccc1Br>>O=[N+]([O-])c1ccccc1O"),
            _reaction_keys("Cc1cccc(B(O)O)c1>>Cc1cccc(Br)c1"),
        }
        self.assertEqual(got, expected)

    def test_single_reused_id_routes_by_structure(self):
        """A step references a lone id that two live molecules share.

        S1 splits the biaryl into an idle phenyl-bromide leaf (id 20) and the
        tolyl branch. S2 edits the tolyl branch and its methoxy oxygen *reuses*
        id 20, so now both the idle leaf and the tolyl intermediate carry id 20.
        S3 then edits the tolyl intermediate through id 20 alone. The leaf sits
        first on the frontier, so routing by id would edit the wrong molecule;
        routing by the step's product structure edits the intended one.
        """
        target = (
            "[cH:1]1[cH:2][cH:3][cH:4][c:5]([OH:6])[c:7]1"
            "-[c:8]1[cH:9][cH:10][cH:11][cH:12][c:13]1[CH3:14]"
        )
        s1 = (
            [
                {"op": "break_bond", "map_a": 7, "map_b": 8},
                {"op": "add_group", "map_idx": 7, "fragment_smiles": "*Br"},
                {"op": "add_group", "map_idx": 8, "fragment_smiles": "*Br"},
            ],
            target + ">>"
            "[cH:1]1[cH:2][cH:3][cH:4][c:5]([OH:6])[c:7]1[Br:20]."
            "[c:8]1([Br:21])[cH:9][cH:10][cH:11][cH:12][c:13]1[CH3:14]",
        )
        # Tolyl branch, step 1: swap Br for OMe whose oxygen reuses id 20.
        s2 = (
            [
                {"op": "remove_group", "map_idx": 8, "map_root_idx": 21},
                {"op": "add_group", "map_idx": 8, "fragment_smiles": "*OC"},
            ],
            "[c:8]1([Br:21])[cH:9][cH:10][cH:11][cH:12][c:13]1[CH3:14]>>"
            "[c:8]1([O:20][CH3:22])[cH:9][cH:10][cH:11][cH:12][c:13]1[CH3:14]",
        )
        # Tolyl branch, step 2: turn that OMe oxygen (id 20) into a thioether.
        s3 = (
            [{"op": "change_atom", "map_idx": 20, "symbol": "S"}],
            "[c:8]1([O:20][CH3:22])[cH:9][cH:10][cH:11][cH:12][c:13]1[CH3:14]>>"
            "[c:8]1([S:20][CH3:22])[cH:9][cH:10][cH:11][cH:12][c:13]1[CH3:14]",
        )

        idle_leaf = _mol(_strip(s1[1].split(">>", 1)[1].split(".")[0]), in_stock=True)
        s3_leaf = _mol(_strip(s3[1].split(">>", 1)[1]), in_stock=True)
        s3_prod = _mol(
            _strip(s3[1].split(">>", 1)[0]),
            children=[_rxn(s3[0], s3[1], children=[s3_leaf])],
        )
        tolyl = _mol(
            _strip(s2[1].split(">>", 1)[0]),
            children=[_rxn(s2[0], s2[1], children=[s3_prod])],
        )
        tree = _mol(
            _strip(target),
            children=[_rxn(s1[0], s1[1], children=[idle_leaf, tolyl])],
        )

        plan = tree_to_op_plan(tree)
        rendered = render_plan(plan, strip_frontier_maps=True)
        by_id = {e["id"]: e for e in rendered["route"]}

        # S3 must edit the tolyl methoxy (OMe -> SMe), not the idle phenyl
        # bromide that also carries id 20.
        self.assertEqual(
            _reaction_keys(by_id["S3"]["smiles"]),
            _reaction_keys("COc1ccccc1C>>CSc1ccccc1C"),
        )

    def test_intra_step_add_group_then_add_bond(self):
        """A ring-closing add_bond onto an atom introduced earlier in the step.

        The dithiolane fragment carries explicit maps :98/:99; a subsequent
        add_bond closes onto :99. The frontier-owner check must not demand the
        product already contain 99 (it is created mid-step).
        """
        steps = [
            (
                [
                    {"op": "change_bond_order", "map_a": 2, "map_b": 3, "delta": -1},
                    {"op": "remove_group", "map_idx": 2, "map_root_idx": 3},
                    {"op": "add_group", "map_idx": 2, "fragment_smiles": "*[S:98]CC[S:99]"},
                    {"op": "add_bond", "map_a": 2, "map_b": 99},
                ],
                "[CH3:1][C:2](=[O:3])[CH3:4]>>C1C[S:99][C:2]([CH3:1])([CH3:4])[S:98]1",
            ),
        ]
        tree = _linear_tree(steps)
        plan = tree_to_op_plan(tree)
        rendered = render_plan(plan, strip_frontier_maps=True)

        self.assertEqual(len(rendered["route"]), 1)
        self.assertEqual(
            _reaction_keys(rendered["route"][0]["smiles"]),
            _reaction_keys("CC(C)=O>>C1CSC(C)(C)S1"),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
