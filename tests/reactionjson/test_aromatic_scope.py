"""Characterization of ReactionJSON's scope on aromatic chemistry.

These tests pin down what the ten-operation vocabulary can and cannot express
for classical aromatization and dearomatization reactions. They assert the
*current* behaviour deliberately: the point is that the boundary is documented
and regression-guarded, not that every reaction is supported.

Summary of the boundary, all verified below:

* Retrosynthesis operates on the product, so what matters is whether the
  *product* is aromatic, not whether the reaction is called an aromatization.
* Dearomatization reactions have a non-aromatic product. The precursor arene is
  reached by raising a bond order, which works.
* Aromatization reactions have an aromatic product. The precursor is reached by
  taking the ring apart, which requires editing an aromatic ring bond. That is
  refused -- see ``test_aromatization_is_uniformly_refused``.
* Substituent chemistry on an intact aromatic ring works, with one asymmetry:
  an exocyclic bond to a neutral aromatic nitrogen needs ``remove_group``
  rather than ``break_bond``.
"""

from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from synthex.reactionjson import Reaction, apply_reaction, get_mapped_smiles

# --------------------------------------------------------------------------
# helpers: locate atoms by environment so the tests do not depend on the
# atom-map numbering that get_mapped_smiles happens to assign
# --------------------------------------------------------------------------


def _mapped(smiles: str) -> tuple[str, Chem.Mol]:
    mapped = get_mapped_smiles(smiles)
    molecule = Chem.MolFromSmiles(mapped)
    assert molecule is not None, f"could not re-parse mapped SMILES {mapped!r}"
    return mapped, molecule


def _bond_maps(molecule: Chem.Mol, smarts: str) -> tuple[int, int]:
    """Atom-map numbers of the first two atoms matching ``smarts``."""

    query = Chem.MolFromSmarts(smarts)
    assert query is not None, f"bad SMARTS {smarts!r}"
    hit = molecule.GetSubstructMatch(query)
    assert len(hit) >= 2, f"{smarts!r} did not match {Chem.MolToSmiles(molecule)!r}"
    return tuple(molecule.GetAtomWithIdx(hit[index]).GetAtomMapNum() for index in (0, 1))


def _fragments(precursors: str) -> list[Chem.Mol]:
    molecules = [Chem.MolFromSmiles(part) for part in precursors.split(".")]
    assert all(m is not None for m in molecules), f"unparseable precursors {precursors!r}"
    return molecules


def _formulas(precursors: str) -> set[str]:
    return {rdMolDescriptors.CalcMolFormula(m) for m in _fragments(precursors)}


def _radical_electrons(precursors: str) -> int:
    return sum(
        atom.GetNumRadicalElectrons() for m in _fragments(precursors) for atom in m.GetAtoms()
    )


def _apply(mapped: str, operations: list[dict]) -> str:
    return apply_reaction(Reaction(product=mapped, operations=operations))


# --------------------------------------------------------------------------
# dearomatization: non-aromatic product, arene precursor -- supported
# --------------------------------------------------------------------------

DEAROMATIZATIONS = [
    # name,                    product,          bond to raise, expected precursor
    ("Birch reduction", "C1=CC=CCC1", "[CH2]~[CH2]", "C6H6"),
    ("pyridine hydrogenation", "C1CCNCC1", "[CH2]~[NH]", "C5H9N"),
    ("phenol dearomatization", "O=C1CCCC=C1", "[CH2]~[CH2]", "C6H6O"),
]


@pytest.mark.parametrize(("name", "product", "smarts", "expected"), DEAROMATIZATIONS)
def test_dearomatization_reaches_the_arene_precursor(name, product, smarts, expected):
    """A non-aromatic product can be traced back to its aromatic precursor."""

    mapped, molecule = _mapped(product)
    map_a, map_b = _bond_maps(molecule, smarts)

    precursors = _apply(
        mapped, [{"op": "change_bond_order", "map_a": map_a, "map_b": map_b, "delta": 1}]
    )

    assert expected in _formulas(precursors), name
    assert _radical_electrons(precursors) == 0, f"{name} produced a radical species"


# --------------------------------------------------------------------------
# aromatization: aromatic product, ring must come apart -- NOT supported
# --------------------------------------------------------------------------

AROMATIZATIONS = [
    ("Paal-Knorr pyrrole", "Cc1ccc(C)[nH]1", "[nH]~[cH0]"),
    ("Debus-Radziszewski imidazole", "c1cnc[nH]1", "[nH]~[cH]"),
    ("Fischer indole", "Cc1cc2ccccc2[nH]1", "[nH]~[cH0]"),
    ("Hantzsch pyridine", "Cc1cccc(C)n1", "[n]~[cH0]"),
    ("Knorr pyrazole", "Cc1cc(C)[nH]n1", "[nH]~[n]"),
    ("Skraup quinoline", "c1ccc2ncccc2c1", "[n]~[cH]"),
    ("alkyne cyclotrimerization", "c1ccccc1", "[cH]~[cH]"),
    ("Scholl ring fusion", "c1ccc2ccccc2c1", "[cH0]~[cH0]"),
]


@pytest.mark.parametrize(("name", "product", "smarts"), AROMATIZATIONS)
def test_aromatization_is_uniformly_refused(name, product, smarts):
    """Taking an aromatic ring apart is refused, and refused loudly.

    Aromatic bond order is not integral, so no integer ``delta`` describes it and
    ``break_bond`` would have to invent hydrogens. The important property is that
    every one of these fails with a ValueError rather than silently returning a
    wrong precursor -- an aromatization reaction can never be quietly mis-encoded.
    """

    mapped, molecule = _mapped(product)
    map_a, map_b = _bond_maps(molecule, smarts)

    with pytest.raises(ValueError):
        _apply(mapped, [{"op": "break_bond", "map_a": map_a, "map_b": map_b}])
    with pytest.raises(ValueError):
        _apply(mapped, [{"op": "change_bond_order", "map_a": map_a, "map_b": map_b, "delta": -1}])


# --------------------------------------------------------------------------
# substituent chemistry on an intact ring -- supported
# --------------------------------------------------------------------------

SUBSTITUENT_REACTIONS = [
    ("Suzuki coupling", "c1ccccc1-c1ccccc1", "[cH0]-[cH0]", {"C6H6"}),
    ("Friedel-Crafts acylation", "CC(=O)c1ccccc1", "[cH0]-[CX3]=O", {"C6H6", "C2H4O"}),
    ("Buchwald-Hartwig N-arylation", "c1ccccc1Nc1ccccc1", "[cH0]-[NX3]", {"C6H6", "C6H7N"}),
    ("aryl ether cleavage", "COc1ccccc1", "[CH3]-[OX2]", {"CH4", "C6H6O"}),
    ("Ullmann biaryl", "Cc1ccc(-c2ccccc2)cc1", "[cH0]-[cH0]", {"C6H6", "C7H8"}),
]


@pytest.mark.parametrize(("name", "product", "smarts", "expected"), SUBSTITUENT_REACTIONS)
def test_exocyclic_disconnection_keeps_the_ring_intact(name, product, smarts, expected):
    """An exocyclic bond to aromatic carbon or oxygen disconnects normally."""

    mapped, molecule = _mapped(product)
    map_a, map_b = _bond_maps(molecule, smarts)

    precursors = _apply(mapped, [{"op": "break_bond", "map_a": map_a, "map_b": map_b}])

    assert _formulas(precursors) == expected, name
    assert _radical_electrons(precursors) == 0, f"{name} produced a radical species"


# --------------------------------------------------------------------------
# the break_bond / remove_group asymmetry on aromatic nitrogen
# --------------------------------------------------------------------------

N_SUBSTITUTED_AZOLES = [
    ("1-methylimidazole", "Cn1ccnc1", "C3H4N2"),
    ("N-methylpyrrole", "Cn1cccc1", "C4H5N"),
    ("N-methylindole", "Cn1ccc2ccccc21", "C8H7N"),
    ("N-methylpyrazole", "Cn1cccn1", "C3H4N2"),
]


def _azole_maps(product: str) -> tuple[str, int, int]:
    """Return (mapped, nitrogen_map, methyl_map) for an N-methylated azole."""

    mapped, molecule = _mapped(product)
    methyl = next(
        atom.GetAtomMapNum()
        for atom in molecule.GetAtoms()
        if atom.GetSymbol() == "C" and not atom.GetIsAromatic() and atom.GetTotalNumHs() == 3
    )
    nitrogen = next(
        atom.GetAtomMapNum()
        for atom in molecule.GetAtoms()
        if atom.GetSymbol() == "N"
        and atom.GetIsAromatic()
        and any(n.GetAtomMapNum() == methyl for n in atom.GetNeighbors())
    )
    return mapped, nitrogen, methyl


@pytest.mark.parametrize(("name", "product", "expected"), N_SUBSTITUTED_AZOLES)
def test_break_bond_cannot_dealkylate_a_neutral_azole_nitrogen(name, product, expected):
    """``break_bond`` does not refill the freed nitrogen's valence.

    Removing the substituent leaves an aromatic nitrogen needing one hydrogen to
    stay valid. ``break_bond`` does not add it, so sanitization fails. This is
    the documented reason N-alkylation and N-arylation of azoles -- extremely
    common chemistry -- must use ``remove_group`` instead.
    """

    mapped, nitrogen, methyl = _azole_maps(product)

    with pytest.raises(ValueError):
        _apply(mapped, [{"op": "break_bond", "map_a": nitrogen, "map_b": methyl}])


@pytest.mark.parametrize(("name", "product", "expected"), N_SUBSTITUTED_AZOLES)
def test_remove_group_dealkylates_a_neutral_azole_nitrogen(name, product, expected):
    """``remove_group`` refills the anchor's valence, so the azole is recovered.

    Note the asymmetry with ``break_bond``: the departing group is deleted rather
    than returned as a second precursor, so the alkylating agent must be recorded
    in the step's conditions instead.
    """

    mapped, nitrogen, methyl = _azole_maps(product)

    precursors = _apply(
        mapped, [{"op": "remove_group", "map_idx": nitrogen, "map_root_idx": methyl}]
    )

    assert _formulas(precursors) == {expected}, name
    assert _radical_electrons(precursors) == 0, f"{name} produced a radical species"


@pytest.mark.parametrize(("name", "product", "expected"), N_SUBSTITUTED_AZOLES)
def test_remove_group_argument_order_silently_discards_the_ring(name, product, expected):
    """Swapping ``map_idx`` and ``map_root_idx`` returns the wrong molecule quietly.

    ``map_idx`` is the anchor that stays and ``map_root_idx`` roots the group that
    is deleted. Reversing them keeps the methyl and deletes the whole heterocycle,
    which is chemically absurd but a perfectly valid molecule, so nothing raises.
    This test exists to document the hazard, not to endorse it.
    """

    mapped, nitrogen, methyl = _azole_maps(product)

    precursors = _apply(
        mapped, [{"op": "remove_group", "map_idx": methyl, "map_root_idx": nitrogen}]
    )

    assert _formulas(precursors) == {"CH4"}, f"{name}: expected the ring to be discarded"


# --------------------------------------------------------------------------
# a cationic aromatic nitrogen needs no extra hydrogen
# --------------------------------------------------------------------------


def test_break_bond_dealkylates_a_cationic_azolium_nitrogen():
    """An imidazolium nitrogen keeps a valid valence, so ``break_bond`` works.

    This is why the charged N-aryl imidazolium ligands disconnect with a plain
    ``break_bond`` while their neutral analogues do not.
    """

    mapped, molecule = _mapped("C[n+]1ccn(C)c1")
    map_a, map_b = _bond_maps(molecule, "[CH3]-[n+]")

    precursors = _apply(mapped, [{"op": "break_bond", "map_a": map_a, "map_b": map_b}])

    assert _formulas(precursors) == {"CH4", "C4H7N2+"}
    assert _radical_electrons(precursors) == 0


# --------------------------------------------------------------------------
# error messages must name a reachable alternative, not a dead end
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "product", "expected"), N_SUBSTITUTED_AZOLES)
def test_break_bond_error_points_at_remove_group(name, product, expected):
    """The failure a model actually hits must name the operation that works."""

    mapped, nitrogen, methyl = _azole_maps(product)

    with pytest.raises(ValueError, match="remove_group") as caught:
        _apply(mapped, [{"op": "break_bond", "map_a": nitrogen, "map_b": methyl}])
    assert "aromatic N" in str(caught.value)


@pytest.mark.parametrize("product", ["Cc1ccccc1", "COc1ccccc1", "C[n+]1ccn(C)c1"])
def test_no_remove_group_hint_on_a_successful_disconnection(product):
    """The hint must not appear where break_bond works. Guards over-triggering."""

    mapped, molecule = _mapped(product)
    smarts = "[CH3]-[n+]" if "n+" in mapped else "[CH3]-[cH0,OX2]"
    map_a, map_b = _bond_maps(molecule, smarts)

    precursors = _apply(mapped, [{"op": "break_bond", "map_a": map_a, "map_b": map_b}])
    assert _radical_electrons(precursors) == 0


def test_aromatic_errors_do_not_advise_an_unreachable_operation():
    """Neither aromatic refusal may tell the caller to kekulize or to reduce first.

    ``change_bond_order`` used to say "kekulize first" and ``break_bond`` used to
    redirect to ``change_bond_order``. Both were dead ends: the vocabulary has no
    kekulize operation and ``change_bond_order`` rejects aromatic bonds too.
    """

    mapped, molecule = _mapped("c1ccccc1")
    map_a, map_b = _bond_maps(molecule, "[cH]~[cH]")

    with pytest.raises(ValueError) as broke:
        _apply(mapped, [{"op": "break_bond", "map_a": map_a, "map_b": map_b}])
    with pytest.raises(ValueError) as reduced:
        _apply(mapped, [{"op": "change_bond_order", "map_a": map_a, "map_b": map_b, "delta": -1}])

    for message in (str(broke.value), str(reduced.value)):
        assert "kekulize first" not in message
        assert "exocyclic" in message
    assert "Use change_bond_order to reduce" not in str(broke.value)
