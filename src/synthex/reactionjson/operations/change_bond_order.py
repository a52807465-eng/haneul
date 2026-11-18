"""Atomic op: change_bond_order.

Changes the order of an existing single/double/triple bond by a signed delta.
Implicit hydrogens on the two endpoints are adjusted so that the new bond order
does not create a valence violation (or, on order decrease, so the atoms get
back the Hs they had freed).

Aromatic bonds are rejected: callers must kekulize first. To remove a bond
entirely (result order 0), callers must use `break_bond` instead.
"""

from rdkit import Chem
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec

_ORDER_MAP = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}


def change_bond_order(mol: Chem.Mol, idx_a: int, idx_b: int, delta: int) -> Chem.Mol:
    """Change bond order between idx_a and idx_b by delta (+/- 1/2/3).

    Single -> Double is delta +1. Triple -> Single is delta -2. Etc.

    Returns a new sanitized Mol (input not mutated).

    Raises ValueError if:
        - delta is not in {-3, -2, -1, +1, +2, +3}
        - either index is out of range
        - no bond exists between the two atoms
        - the bond is aromatic (kekulize first)
        - the result would be order 0 (use break_bond) or > 3
        - the resulting molecule fails sanitization (valence violation)
    """
    if delta not in (-3, -2, -1, 1, 2, 3):
        raise ValueError(f"delta must be in {{-3,-2,-1,+1,+2,+3}}; got {delta!r}")

    n = mol.GetNumAtoms()
    if not (0 <= idx_a < n) or not (0 <= idx_b < n):
        raise ValueError(f"atom index out of range: idx_a={idx_a}, idx_b={idx_b}, num_atoms={n}")

    bond = mol.GetBondBetweenAtoms(idx_a, idx_b)
    if bond is None:
        raise ValueError(f"no bond exists between atoms {idx_a} and {idx_b}")

    # Reject aromatic bonds: GetBondTypeAsDouble() returns 1.5, which would
    # silently int-cast to 1. Check the BondType directly.
    if bond.GetBondType() == Chem.BondType.AROMATIC:
        raise ValueError(
            "cannot change the order of an aromatic bond: aromatic bond order is "
            "not integral, and reducing it would require adding hydrogens this "
            "operation cannot infer. No operation kekulizes, so an aromatic ring "
            "cannot be opened; disconnect at an exocyclic single bond instead."
        )

    current = int(bond.GetBondTypeAsDouble())
    if current not in _ORDER_MAP:
        raise ValueError(
            f"bond between {idx_a} and {idx_b} has unsupported type "
            f"{bond.GetBondType()}; only SINGLE/DOUBLE/TRIPLE are supported"
        )

    new_order = current + delta
    if new_order <= 0:
        raise ValueError(
            f"delta={delta} would reduce bond order from {current} to "
            f"{new_order}; use break_bond to remove the bond"
        )
    if new_order > 3:
        raise ValueError(
            f"delta={delta} would raise bond order from {current} to "
            f"{new_order}; maximum supported order is 3"
        )

    rw = Chem.RWMol(mol)

    # Adjust implicit-H counts on both endpoints. When increasing bond order,
    # each endpoint loses |delta| Hs; when decreasing, each gains |delta| Hs.
    # RDKit's auto-adjust on SetBondType is unreliable when atoms carry explicit
    # H counts or have noImplicit set, so we handle both paths manually.
    for idx in (idx_a, idx_b):
        atom = rw.GetAtomWithIdx(idx)
        if atom.GetNumExplicitHs() > 0 or atom.GetNoImplicit():
            new_h = atom.GetNumExplicitHs() - delta
            if new_h < 0:
                new_h = 0
            atom.SetNumExplicitHs(new_h)
            # Keep noImplicit as-is; caller-provided state is preserved.
        else:
            # Implicit-H atom: let sanitize recompute from scratch.
            atom.SetNoImplicit(False)
            atom.SetNumExplicitHs(0)

    new_bond = rw.GetBondBetweenAtoms(idx_a, idx_b)
    new_bond.SetBondType(_ORDER_MAP[new_order])
    # Clear E/Z on the edited bond AND the `/`/`\` direction markers on all
    # bonds adjacent to either endpoint: AssignStereochemistry re-perceives
    # double-bond geometry from those directional singles, so leaving them in
    # place resurrects stale E/Z after a reduce+raise round-trip.
    new_bond.SetStereo(Chem.BondStereo.STEREONONE)
    new_bond.SetBondDir(Chem.BondDir.NONE)
    for end_idx in (idx_a, idx_b):
        for nbr_bond in rw.GetAtomWithIdx(end_idx).GetBonds():
            nbr_bond.SetBondDir(Chem.BondDir.NONE)

    new_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"change_bond_order({idx_a}, {idx_b}, delta={delta}) produced an "
            f"invalid molecule: {exc}"
        ) from exc

    # Changing a bond order may create/destroy a stereocenter (e.g. C=C -> C-C
    # clears any E/Z that was on this bond, or a new sp2 center appears).
    # SanitizeMol does not re-perceive stereo, so force a fresh pass.
    Chem.AssignStereochemistry(new_mol, cleanIt=True, force=True)

    return new_mol


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════
