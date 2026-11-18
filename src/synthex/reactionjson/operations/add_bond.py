"""Atomic op: add_bond.

Adds a single/double/triple bond between two atoms in a molecule, returning a new
sanitized Mol. Implicit hydrogens on the two endpoints are decremented as needed
so that the new bond does not create a valence violation.
"""

from rdkit import Chem
from rdkit.Chem import AllChem  # noqa: F401 - available per the op spec

_ORDER_MAP = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}

_ORDER_DELTA = {1: 1, 2: 2, 3: 3}


def add_bond(mol: Chem.Mol, idx_a: int, idx_b: int, order: int = 1) -> Chem.Mol:
    """Add a bond between atoms idx_a and idx_b of order 1, 2, or 3.

    Returns a new sanitized Mol (input is not mutated).

    Raises ValueError if:
        - order is not in {1, 2, 3}
        - idx_a == idx_b
        - either index is out of range
        - a bond already exists between the two atoms
        - the resulting molecule fails sanitization (e.g. valence violation)
    """
    if order not in _ORDER_MAP:
        raise ValueError(f"order must be 1, 2, or 3; got {order!r}")
    if idx_a == idx_b:
        raise ValueError(f"idx_a and idx_b must differ (got {idx_a})")

    n = mol.GetNumAtoms()
    if not (0 <= idx_a < n) or not (0 <= idx_b < n):
        raise ValueError(f"atom index out of range: idx_a={idx_a}, idx_b={idx_b}, num_atoms={n}")

    if mol.GetBondBetweenAtoms(idx_a, idx_b) is not None:
        raise ValueError(f"bond already exists between atoms {idx_a} and {idx_b}")

    rw = Chem.RWMol(mol)
    delta = _ORDER_DELTA[order]

    # RDKit will NOT auto-reduce implicit Hs when you call AddBond on atoms that
    # have "noImplicit" semantics or when explicit Hs are set. To be robust, we
    # explicitly decrement either numExplicitHs (if set) or rely on the implicit
    # H calculation by updating the property cache after the edit. In practice:
    # if an atom was built by SMILES parsing (e.g. "C"), it has numExplicitHs=0
    # and its H count is implicit-from-valence. Decrementing explicit Hs when
    # they are already zero would fail, so we only touch explicit Hs when set.
    for idx in (idx_a, idx_b):
        atom = rw.GetAtomWithIdx(idx)
        n_expl = atom.GetNumExplicitHs()
        if n_expl >= delta:
            atom.SetNumExplicitHs(n_expl - delta)
        elif n_expl > 0:
            # Partial: zero out explicit Hs, rest will come from implicit recompute.
            atom.SetNumExplicitHs(0)
        # else: rely on implicit-H recompute during sanitization.

    rw.AddBond(idx_a, idx_b, _ORDER_MAP[order])

    new_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
    except (Chem.AtomValenceException, Chem.KekulizeException, ValueError) as exc:
        raise ValueError(
            f"add_bond({idx_a}, {idx_b}, order={order}) produced an invalid molecule: {exc}"
        ) from exc

    # SanitizeMol does not re-perceive stereochemistry; the new bond may
    # create or destroy a stereocenter elsewhere, so force a fresh pass.
    Chem.AssignStereochemistry(new_mol, cleanIt=True, force=True)

    return new_mol


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════
