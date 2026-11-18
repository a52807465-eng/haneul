"""Atomic op: set_explicit_h.

Sets the number of explicit hydrogens on a single atom to exactly ``n`` and
locks the H count (``noImplicit=True``) so later sanitization will not silently
re-add or remove Hs. Returns a new sanitized Mol. Raises ValueError on any
out-of-range index, negative n, or valence violation.
"""

from rdkit import Chem
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec


def set_explicit_h(mol: Chem.Mol, idx: int, n: int) -> Chem.Mol:
    """Set the number of explicit hydrogens on atom idx to exactly n.

    Returns a new sanitized Mol (input is not mutated). The target atom has
    ``NoImplicit=True`` set after this op, so subsequent edits will not silently
    adjust its H count.

    Raises ValueError if:
        - idx is out of range
        - n < 0
        - the resulting molecule fails sanitization (e.g. valence violation)
    """
    num_atoms = mol.GetNumAtoms()
    if not (0 <= idx < num_atoms):
        raise ValueError(f"atom index out of range: idx={idx}, num_atoms={num_atoms}")
    if n < 0:
        raise ValueError(f"n must be >= 0; got {n}")

    rw = Chem.RWMol(mol)

    # Kekulize before touching an aromatic atom: changing the H count on an
    # aromatic ring atom can invalidate the aromatic perception, which then
    # fails kekulization on the way back out. Kekulizing up-front makes the
    # bond orders explicit so sanitization can re-perceive aromaticity cleanly.
    atom = rw.GetAtomWithIdx(idx)
    if atom.GetIsAromatic():
        Chem.Kekulize(rw, clearAromaticFlags=True)
        atom = rw.GetAtomWithIdx(idx)

    atom.SetNumExplicitHs(n)
    atom.SetNoImplicit(True)

    new_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"set_explicit_h(idx={idx}, n={n}) produced an invalid molecule: {exc}"
        ) from exc

    # SanitizeMol does NOT re-perceive stereochemistry; changing H count can
    # promote/demote a stereocenter (e.g. CHRR' vs CH2R), so force a fresh pass.
    Chem.AssignStereochemistry(new_mol, cleanIt=True, force=True)

    return new_mol


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════
