"""Atomic op: clear_stereocenter.

Clears the chiral tag at a single atom (sets it to CHI_UNSPECIFIED), returning
a new sanitized Mol. Idempotent: calling on an already-unspecified atom is a
no-op. Stereochemistry elsewhere in the molecule is preserved.
"""

from rdkit import Chem
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec


def clear_stereocenter(mol: Chem.Mol, idx: int) -> Chem.Mol:
    """Clear stereochemistry at atom idx (set to CHI_UNSPECIFIED). Idempotent.

    Returns a new sanitized Mol (input is not mutated).

    Raises ValueError only if idx is out of range.
    """
    n = mol.GetNumAtoms()
    if not (0 <= idx < n):
        raise ValueError(f"atom index out of range: idx={idx}, num_atoms={n}")

    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(idx).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)

    new_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(f"clear_stereocenter({idx}) produced an invalid molecule: {exc}") from exc

    # SanitizeMol does NOT re-perceive stereochemistry. Without this pass the
    # cached CIP codes/stereo flags survive and the @/@@ marker stays in the
    # canonical SMILES even though the ChiralTag is unspecified.
    Chem.AssignStereochemistry(new_mol, cleanIt=True, force=True)

    return new_mol


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════
