"""Atomic op: invert_stereocenter.

Flips the chiral tag (CW <-> CCW) at a tetrahedral stereocenter, returning a
new sanitized Mol with re-perceived stereochemistry. Only atoms that already
carry a tetrahedral chiral tag are accepted; unspecified or non-tetrahedral
chirality is out of scope for this op.
"""

from rdkit import Chem
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec

_FLIP = {
    Chem.ChiralType.CHI_TETRAHEDRAL_CW: Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW: Chem.ChiralType.CHI_TETRAHEDRAL_CW,
}


def invert_stereocenter(mol: Chem.Mol, idx: int) -> Chem.Mol:
    """Invert the chiral tag at atom idx (CW <-> CCW).

    Returns a new sanitized Mol (input is not mutated) with stereo re-perceived
    so that downstream CIP labels (R/S via FindMolChiralCenters) reflect the
    inversion.

    Raises ValueError if:
        - idx is out of range
        - the atom has CHI_UNSPECIFIED (not a stereocenter)
        - the atom has CHI_OTHER or any non-tetrahedral chirality
        - the resulting molecule fails sanitization
    """
    n = mol.GetNumAtoms()
    if not (0 <= idx < n):
        raise ValueError(f"atom index out of range: idx={idx}, num_atoms={n}")

    tag = mol.GetAtomWithIdx(idx).GetChiralTag()
    if tag == Chem.ChiralType.CHI_UNSPECIFIED:
        raise ValueError(
            f"atom {idx} has CHI_UNSPECIFIED: not a stereocenter. Set a chiral "
            f"tag first or use a different op."
        )
    if tag not in _FLIP:
        raise ValueError(
            f"atom {idx} has non-tetrahedral chirality {tag!r}; out of scope "
            f"for invert_stereocenter."
        )

    rw = Chem.RWMol(mol)
    rw.GetAtomWithIdx(idx).SetChiralTag(_FLIP[tag])

    new_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(f"invert_stereocenter({idx}) produced an invalid molecule: {exc}") from exc

    # The CW/CCW tag is relative to the neighbor order in the current SMILES
    # traversal; after flipping, round-trip through canonical SMILES so the
    # tag is re-normalized against the canonical atom ordering.
    canon = Chem.MolToSmiles(new_mol)
    new_mol = Chem.MolFromSmiles(canon)
    if new_mol is None:
        raise ValueError(f"invert_stereocenter({idx}): canonical round-trip failed for {canon!r}")

    # SanitizeMol does NOT re-perceive stereochemistry; force a fresh pass so
    # that CIP labels (R/S) on FindMolChiralCenters reflect the inversion.
    Chem.AssignStereochemistry(new_mol, cleanIt=True, force=True)

    return new_mol


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════
