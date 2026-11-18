"""Atom and stereo accounting must report imbalance without ruling on it.

The Critic prompt asks the model to audit atom counts and stereocentres, and
models claim to have done so while passing precursors that drop a fragment or
flatten a centre. Computing it here turns an ignored instruction into evidence.
"""

from __future__ import annotations

from synthex.routejson import atom_deficit, atom_stereo_accounting


def test_a_balanced_step_reports_no_imbalance():
    block = atom_stereo_accounting("CCCC", "CC.CC")

    assert "MISSING from precursors (present in product): none" in block
    assert "SURPLUS in precursors (lost in the forward step): none" in block
    assert "Product heavy atoms:    4" in block


def test_a_leaving_group_shows_as_surplus():
    """A bromide lost going forward is a legitimate surplus, not an error."""
    block = atom_stereo_accounting("CC1OCC2COC1(C)O2", "CC(Br)C1(C)OCC(CO)O1")

    assert "SURPLUS in precursors (lost in the forward step): 1x Br" in block
    assert "MISSING from precursors (present in product): none" in block


def test_a_condensation_byproduct_shows_as_surplus():
    """Ketalization loses water going forward, so precursors carry one extra O."""
    block = atom_stereo_accounting("CC(Br)C1(C)OCC(CO)O1", "CC(=O)C(C)Br.OCC(O)CO")

    assert "SURPLUS in precursors (lost in the forward step): 1x O" in block


def test_a_dropped_fragment_shows_as_missing():
    """The dangerous direction: the product has atoms the precursors cannot supply."""
    block = atom_stereo_accounting("CCCCCC", "CC")

    assert "MISSING from precursors (present in product): 4x C" in block
    assert atom_deficit("CCCCCC", "CC") == {"C": 4}


def test_several_elements_missing_at_once_signals_a_whole_fragment():
    block = atom_stereo_accounting("CC(=O)Nc1ccccc1", "CC(=O)O")

    missing = [line for line in block.splitlines() if line.startswith("MISSING")][0]
    assert "6x C" in missing and "1x N" in missing


def test_a_flattened_stereocentre_shows_in_the_stereo_delta():
    """Atom counts can balance while stereochemistry is silently lost."""
    block = atom_stereo_accounting("C[C@H](O)CC", "C[CH](O)CC")

    assert "MISSING from precursors (present in product): none" in block
    assert "delta assigned=-1" in block


def test_a_lost_double_bond_geometry_shows_in_the_bond_delta():
    block = atom_stereo_accounting("C/C=C/C", "CC=CC")

    assert "Stereo double bonds  product: 1   precursors: 0   delta=-1" in block


def test_atom_deficit_ignores_surplus():
    assert atom_deficit("CC", "CCCC") == {}
    assert atom_deficit("CCCC", "CC.CC") == {}


def test_unparseable_input_is_reported_not_raised():
    assert "Accounting unavailable" in atom_stereo_accounting("not-a-molecule", "CC")
    assert "Accounting unavailable" in atom_stereo_accounting("CC", "not-a-molecule")
    assert atom_deficit("not-a-molecule", "CC") == {}


def test_no_precursors_is_reported():
    assert "Accounting unavailable" in atom_stereo_accounting("CC", "")
