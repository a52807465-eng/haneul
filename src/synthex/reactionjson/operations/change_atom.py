"""Atomic op: change_atom.

Transmutes the element of a single atom in a molecule (e.g. C -> N), letting
RDKit's sanitizer recompute implicit hydrogens for the new element's default
valence. Returns a new sanitized Mol; the input is not mutated.
"""

from rdkit import Chem
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec


def change_atom(mol: Chem.Mol, idx: int, symbol: str) -> Chem.Mol:
    """Change the element of atom idx to `symbol` (e.g. 'C' -> 'N').

    Returns a new sanitized Mol (input is not mutated).

    Raises ValueError if:
        - idx is out of range
        - symbol is not a known element
        - the resulting molecule fails sanitization (valence violation,
          aromaticity broken, etc.)
    """
    n = mol.GetNumAtoms()
    if not (0 <= idx < n):
        raise ValueError(f"atom index out of range: idx={idx}, num_atoms={n}")

    # RDKit raises RuntimeError (not returns 0) on unknown symbols.
    pt = Chem.GetPeriodicTable()
    try:
        atomic_num = pt.GetAtomicNumber(symbol)
    except RuntimeError as exc:
        raise ValueError(f"unknown element symbol: {symbol!r}") from exc
    if atomic_num <= 0:
        raise ValueError(f"unknown element symbol: {symbol!r}")

    rw = Chem.RWMol(mol)
    atom = rw.GetAtomWithIdx(idx)

    atom.SetAtomicNum(atomic_num)
    # Implicit-H auto-adjust is unreliable across transmutations. Reset the
    # H-related flags so the sanitizer recomputes H count from the new
    # element's default valence.
    atom.SetNumExplicitHs(0)
    atom.SetNoImplicit(False)

    new_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"change_atom({idx}, {symbol!r}) produced an invalid molecule: {exc}"
        ) from exc

    # SanitizeMol does not re-perceive stereochemistry; transmutation can
    # create or destroy stereocenters, so force a fresh pass.
    Chem.AssignStereochemistry(new_mol, cleanIt=True, force=True)

    return new_mol


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════
