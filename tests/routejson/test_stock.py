"""Stock membership must be exact InChI-key lookup, with no size shortcut.

``load_stock_check`` previously returned True for any molecule with fewer than ten
heavy atoms, before consulting the stock at all. That makes a small closed
building-block set unmeasurable: every permitted block in an industrial
raw-material set is itself under ten heavy atoms, so unrelated small molecules
reported as purchasable and routes "terminated" one step below the target.
"""

from __future__ import annotations

import pytest
from rdkit import Chem
from rdkit.Chem import inchi

from synthex.routejson import load_stock_check

# All well under ten heavy atoms, which is precisely why they used to slip through.
GLYCEROL = "OCC(O)CO"
BENZENE = "c1ccccc1"
ACETIC_ANHYDRIDE = "CC(=O)OC(C)=O"
NITROBENZENE = "O=[N+]([O-])c1ccccc1"


def _stock(tmp_path, *smiles):
    path = tmp_path / "stock.txt"
    path.write_text("\n".join(inchi.MolToInchiKey(Chem.MolFromSmiles(s)) for s in smiles) + "\n")
    return load_stock_check(path)


def test_a_small_molecule_absent_from_the_stock_is_not_in_stock(tmp_path):
    check = _stock(tmp_path, GLYCEROL)

    assert check(GLYCEROL) is True
    for absent in (BENZENE, ACETIC_ANHYDRIDE, NITROBENZENE):
        assert check(absent) is False, f"{absent} is not in the stock file"


def test_an_empty_stock_puts_nothing_in_stock(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("")
    check = load_stock_check(path)

    for smiles in (GLYCEROL, BENZENE, "C", "CC", "O"):
        assert check(smiles) is False, f"{smiles} with an empty stock"


def test_membership_is_independent_of_molecule_size(tmp_path):
    """A large stocked molecule is in stock; a small unstocked one is not."""
    large = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"  # ibuprofen, 15 heavy atoms
    check = _stock(tmp_path, large)

    assert check(large) is True
    assert check("CO") is False


def test_the_module_exposes_no_size_cutoff():
    """Guards against the shortcut being reintroduced under any name."""
    import synthex.routejson.stock as module

    assert not [name for name in vars(module) if "CUTOFF" in name.upper()]
    assert not [name for name in vars(module) if "HEAVY" in name.upper()]


def test_unparseable_and_empty_smiles_are_not_in_stock(tmp_path):
    check = _stock(tmp_path, GLYCEROL)

    assert check("") is False
    assert check("not-a-molecule") is False


def test_a_missing_stock_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="stock file not found"):
        load_stock_check(tmp_path / "nope.txt")
