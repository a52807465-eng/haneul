"""Atomic op: set_bond_stereo.

Sets E/Z stereochemistry on a double bond between two atoms. Requires a
reference neighbor on each end (auto-selected if not provided). Returns a new
sanitized Mol with the stereo assignment encoded so that canonical SMILES
output contains the `/` and `\\` bond-direction markers.
"""

from rdkit import Chem
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec

_STEREO_MAP = {
    "E": Chem.BondStereo.STEREOE,
    "Z": Chem.BondStereo.STEREOZ,
}


def set_bond_stereo(
    mol: Chem.Mol,
    idx_a: int,
    idx_b: int,
    stereo: str,
    ref_a: int | None = None,
    ref_b: int | None = None,
) -> Chem.Mol:
    """Set E/Z stereo on a double bond between idx_a and idx_b.

    stereo: 'E' or 'Z' (case-insensitive).
    ref_a: neighbor of idx_a (other than idx_b) used as the reference for
        stereo. If None, the first non-idx_b neighbor of idx_a is used.
    ref_b: same for idx_b.

    Returns a new sanitized Mol (input is not mutated).

    Raises ValueError if:
        - either index is out of range
        - no bond exists between idx_a and idx_b
        - the bond is not DOUBLE
        - stereo is not in {'E', 'e', 'Z', 'z'}
        - a ref index is supplied but is not a neighbor of its endpoint
        - no valid reference neighbor exists on an end (terminal =CH2)
    """
    if not isinstance(stereo, str):
        raise ValueError(f"stereo must be 'E' or 'Z'; got {stereo!r}")
    stereo_norm = stereo.upper()
    if stereo_norm not in _STEREO_MAP:
        raise ValueError(f"stereo must be 'E' or 'Z'; got {stereo!r}")

    n = mol.GetNumAtoms()
    if not (0 <= idx_a < n) or not (0 <= idx_b < n):
        raise ValueError(f"atom index out of range: idx_a={idx_a}, idx_b={idx_b}, num_atoms={n}")

    bond = mol.GetBondBetweenAtoms(idx_a, idx_b)
    if bond is None:
        raise ValueError(f"no bond exists between atoms {idx_a} and {idx_b}")
    if bond.GetBondType() != Chem.BondType.DOUBLE:
        raise ValueError(
            f"set_bond_stereo requires a DOUBLE bond; bond between {idx_a} and "
            f"{idx_b} is {bond.GetBondType()}"
        )

    def _pick_ref(endpoint: int, partner: int, supplied: int | None) -> int:
        atom = mol.GetAtomWithIdx(endpoint)
        neighbor_idxs = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() != partner]
        if supplied is not None:
            if supplied not in neighbor_idxs:
                raise ValueError(
                    f"ref atom {supplied} is not a neighbor of atom {endpoint} "
                    f"(neighbors excluding partner {partner}: {neighbor_idxs})"
                )
            return supplied
        if not neighbor_idxs:
            raise ValueError(
                f"atom {endpoint} has no neighbor other than {partner}; cannot "
                f"define E/Z stereo on a terminal double bond"
            )
        return neighbor_idxs[0]

    chosen_ref_a = _pick_ref(idx_a, idx_b, ref_a)
    chosen_ref_b = _pick_ref(idx_b, idx_a, ref_b)

    rw = Chem.RWMol(mol)
    new_bond = rw.GetBondBetweenAtoms(idx_a, idx_b)
    # SetStereoAtoms must precede SetStereo; STEREOE/STEREOZ are defined
    # *relative to* these two reference atoms.
    new_bond.SetStereoAtoms(chosen_ref_a, chosen_ref_b)
    new_bond.SetStereo(_STEREO_MAP[stereo_norm])

    new_mol = rw.GetMol()
    try:
        # cleanIt=False is critical: AssignStereochemistry(cleanIt=True,
        # force=True) would recompute bond stereo from atomic coordinates or
        # from `/`/`\` bond dirs (neither present here) and wipe the STEREOE/
        # STEREOZ we just set. Plain SanitizeMol preserves it.
        Chem.SanitizeMol(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"set_bond_stereo({idx_a}, {idx_b}, {stereo!r}) produced an invalid molecule: {exc}"
        ) from exc

    # Canonical SMILES emits `/` and `\` from bond-direction flags on the
    # single bonds adjacent to the double bond. SetDoubleBondNeighborDirections
    # writes those flags from the STEREOE/STEREOZ we just set on the double
    # bond, without clobbering it. (SetBondStereoFromDirections goes the wrong
    # way — it reads directions and writes stereo, which would reset us.)
    Chem.SetDoubleBondNeighborDirections(new_mol)

    return new_mol


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════
