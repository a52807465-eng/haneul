"""Atomic op: remove_group.

Removes a pendant group from a molecule. The bond between the anchor atom
`idx` (which stays) and `root_idx` (the first atom of the group to remove) is
broken, and every atom in the fragment rooted at `root_idx` is deleted. The
anchor's freed valence is refilled with implicit Hs via sanitize.

Only true pendant groups are supported: if the (idx, root_idx) bond is in a
ring, breaking it would not disconnect a fragment, and the op rejects it.
"""

from rdkit import Chem
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec

_ANCHOR_TAG = 999


def remove_group(mol: Chem.Mol, idx: int, root_idx: int) -> Chem.Mol:
    """Remove the pendant group rooted at root_idx, anchored at idx.

    Returns a new sanitized Mol (input not mutated). The anchor `idx` stays;
    every atom in the connected component containing `root_idx` (after breaking
    the idx-root_idx bond) is deleted.

    Raises ValueError if:
        - either index is out of range
        - idx == root_idx
        - no bond exists between idx and root_idx
        - the bond between them is in a ring (not a true pendant)
        - the resulting molecule fails sanitization
    """
    n = mol.GetNumAtoms()
    if not (0 <= idx < n) or not (0 <= root_idx < n):
        raise ValueError(f"atom index out of range: idx={idx}, root_idx={root_idx}, num_atoms={n}")
    if idx == root_idx:
        raise ValueError(f"idx and root_idx must be distinct; got {idx}")

    bond = mol.GetBondBetweenAtoms(idx, root_idx)
    if bond is None:
        raise ValueError(f"no bond exists between atoms {idx} and {root_idx}")

    rw = Chem.RWMol(mol)
    # Ring perception must be current for IsInRing to be reliable on an RWMol.
    Chem.GetSSSR(rw)
    rw_bond = rw.GetBondBetweenAtoms(idx, root_idx)
    if rw_bond.IsInRing():
        raise ValueError(f"not a pendant group - bond ({idx}, {root_idx}) is in a ring")

    bond_order = rw_bond.GetBondTypeAsDouble()

    # Persistent anchor: atom indices will shift after RemoveAtom, so tag the
    # anchor with a sentinel atom-map number we can scan for afterward.
    anchor_atom = rw.GetAtomWithIdx(idx)
    prev_map = anchor_atom.GetAtomMapNum()
    anchor_atom.SetAtomMapNum(_ANCHOR_TAG)

    # Break the bond first, then discover which fragment holds root_idx.
    rw.RemoveBond(idx, root_idx)
    frags = Chem.GetMolFrags(rw, asMols=False, sanitizeFrags=False)
    group_frag = None
    for frag in frags:
        if root_idx in frag:
            group_frag = frag
            break
    if group_frag is None:
        raise ValueError(f"internal: root_idx={root_idx} not found in any fragment")

    # Remove in descending index order to avoid reindex invalidation.
    for aidx in sorted(group_frag, reverse=True):
        rw.RemoveAtom(aidx)

    # Locate the anchor by sentinel tag, restore its prior map number.
    new_anchor_idx = None
    for atom in rw.GetAtoms():
        if atom.GetAtomMapNum() == _ANCHOR_TAG:
            new_anchor_idx = atom.GetIdx()
            atom.SetAtomMapNum(prev_map)
            break
    if new_anchor_idx is None:
        raise ValueError("internal: anchor atom lost during removal")

    anchor = rw.GetAtomWithIdx(new_anchor_idx)
    # The removed neighbor was part of the CIP ordering; clear stereo.
    anchor.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    # Valence refill: if the anchor carries explicit Hs (locked), bump it by
    # the removed bond's order to compensate; otherwise let sanitize recompute.
    if anchor.GetNumExplicitHs() > 0 or anchor.GetNoImplicit():
        anchor.SetNumExplicitHs(anchor.GetNumExplicitHs() + int(bond_order))
    else:
        anchor.SetNoImplicit(False)
        anchor.SetNumExplicitHs(0)

    new_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"remove_group({idx}, {root_idx}) produced an invalid molecule: {exc}"
        ) from exc

    Chem.AssignStereochemistry(new_mol, cleanIt=True, force=True)
    return new_mol


# ===========================================================================
# Tests
# ===========================================================================
