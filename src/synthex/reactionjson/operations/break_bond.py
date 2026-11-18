"""Atomic op: break_bond.

Breaks a single (sigma) bond between two atoms. The freed valence on each
endpoint is filled with implicit hydrogens via re-sanitization. If the broken
bond was part of a stereocenter's definition, the chiral tag on the two
endpoints is cleared; stereo elsewhere in the molecule is preserved.

Design decision: this op only supports breaking sigma (single) bonds. To break
a pi bond, first reduce it with `change_bond_order`. Rationale: breaking a pi
bond by itself leaves a sigma bond behind (net order change of 2 -> 1 or 3 ->
2), which is semantically `change_bond_order`, not "break". This avoids
ambiguity about what "break a double bond" means (remove entirely? or reduce?).

Aromatic bonds are a separate case and are rejected outright: their order is not
integral, and `change_bond_order` cannot reduce them either, so an aromatic ring
cannot be taken apart by any operation in the vocabulary.

Freed valence is filled with implicit hydrogens, which is why an exocyclic bond to
a neutral aromatic heteroatom fails: kekulization needs a hydrogen the sanitizer
will not add there. `remove_group` handles that disconnection instead. See
tests/reactionjson/test_aromatic_scope.py for the full characterization.
"""

from rdkit import Chem
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec


def _freed_heteroatom_hint(mol: Chem.Mol, idx_a: int, idx_b: int) -> str:
    """Name `remove_group` when an aromatic heteroatom loses its substituent.

    Breaking an exocyclic bond to an aromatic N leaves it needing one implicit
    hydrogen for the ring to stay kekulizable, and `break_bond` does not add it.
    `remove_group` refills the anchor's valence, so it expresses the same
    disconnection -- notably N-alkylation and N-arylation of azoles.
    """
    for anchor, leaving in ((idx_a, idx_b), (idx_b, idx_a)):
        atom = mol.GetAtomWithIdx(anchor)
        bond = mol.GetBondBetweenAtoms(anchor, leaving)
        if atom.GetIsAromatic() and atom.GetSymbol() != "C" and not bond.IsInRing():
            return (
                f" Atom {anchor} is an aromatic {atom.GetSymbol()} and needs an implicit"
                f" hydrogen once its exocyclic bond is gone, which break_bond does not"
                f" add. Use remove_group with map_idx on atom {anchor} and map_root_idx"
                f" on atom {leaving} to express the same disconnection."
            )
    return ""


def break_bond(mol: Chem.Mol, idx_a: int, idx_b: int) -> Chem.Mol:
    """Break the single (sigma) bond between atoms idx_a and idx_b.

    Fills freed valence with implicit H. Returns a new sanitized Mol (input is
    not mutated). The result may contain multiple disconnected fragments in a
    single Mol object; the caller decides whether to split.

    Only sigma bonds are supported. For a double or triple bond, use
    `change_bond_order` to reduce it first.

    Raises ValueError if:
        - either index is out of range
        - no bond exists between the two atoms
        - the bond exists but is not SINGLE (pi bond present)
    """
    n = mol.GetNumAtoms()
    if not (0 <= idx_a < n) or not (0 <= idx_b < n):
        raise ValueError(f"atom index out of range: idx_a={idx_a}, idx_b={idx_b}, num_atoms={n}")

    bond = mol.GetBondBetweenAtoms(idx_a, idx_b)
    if bond is None:
        raise ValueError(f"no bond exists between atoms {idx_a} and {idx_b}")

    # Aromatic bonds first: they have an implicit pi component and no integral
    # order, and unlike a pi bond there is no operation that can reduce them.
    if bond.GetBondType() == Chem.BondType.AROMATIC:
        # change_bond_order also rejects aromatic bonds, so pointing there would
        # be a dead end: no operation kekulizes. Name the real alternative.
        raise ValueError(
            f"break_bond does not support aromatic bonds; the bond between {idx_a} "
            f"and {idx_b} is part of an aromatic ring, whose bond order is not "
            f"integral. change_bond_order rejects it for the same reason and no "
            f"operation kekulizes, so an aromatic ring cannot be taken apart. "
            f"Disconnect at an exocyclic single bond instead, or choose a strategy "
            f"that leaves the ring intact."
        )
    # Only sigma bonds: reject anything that isn't a plain single bond.
    if bond.GetBondType() != Chem.BondType.SINGLE:
        raise ValueError(
            f"break_bond only supports sigma bonds; bond between {idx_a} and "
            f"{idx_b} is {bond.GetBondType()}. Use change_bond_order to reduce "
            f"a pi bond first."
        )

    rw = Chem.RWMol(mol)

    # Clear any explicit H counts that were previously locked in: after
    # removing a bond, RDKit will not automatically bump implicit H unless the
    # atom's explicit-H count is "unset" in the sense of needing a recompute.
    # We zero out the noImplicit flag so the sanitizer refills H counts.
    for idx in (idx_a, idx_b):
        atom = rw.GetAtomWithIdx(idx)
        atom.SetNoImplicit(False)
        # Clear chirality on endpoints: once the bond is gone, the original
        # CIP ordering used to assign @/@@ is no longer valid.
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    rw.RemoveBond(idx_a, idx_b)

    new_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"break_bond({idx_a}, {idx_b}) produced an invalid molecule: {exc}."
            + _freed_heteroatom_hint(mol, idx_a, idx_b)
        ) from exc

    # SanitizeMol alone does NOT re-perceive stereochemistry: the stereo
    # property cache carries over from the input, so FindMolChiralCenters will
    # still report the endpoints as stereocenters even though we cleared their
    # ChiralTag. Force a fresh pass.
    Chem.AssignStereochemistry(new_mol, cleanIt=True, force=True)

    return new_mol


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════
