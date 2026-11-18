"""Deterministic atom and stereo accounting for a single retrosynthetic step.

The Critic prompt has always *asked* the model to count atoms and audit
stereocentres. Models reliably claim to have done so while passing precursors
that are missing a fragment or have an inverted centre. Counting here with RDKit
and handing the model the arithmetic turns an instruction it ignores into
evidence it has to explain.

The accounting deliberately reports the imbalance rather than ruling on it. A
deficit can be legitimate -- an oxidant contributing O, a halogen from NBS -- and
a surplus usually is: leaving groups and condensation byproducts are lost going
forward. The verdict stays with the Critic; this evidence is computed independently
for each route step.
"""

from __future__ import annotations

from typing import Any

from rdkit import Chem


def _heavy_atom_counts(molecule: Any) -> dict[str, int]:
    """Element -> count of heavy atoms, hydrogens excluded."""

    counts: dict[str, int] = {}
    for atom in molecule.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == "H":
            continue
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def _stereo_counts(molecule: Any) -> tuple[int, int, int]:
    """Return (assigned centres, unassigned centres, stereo double bonds)."""

    try:
        centres = Chem.FindMolChiralCenters(
            molecule, includeUnassigned=True, useLegacyImplementation=False
        )
    except Exception:
        centres = []
    assigned = sum(1 for _, tag in centres if tag != "?")
    unassigned = sum(1 for _, tag in centres if tag == "?")
    bonds = sum(1 for bond in molecule.GetBonds() if bond.GetStereo() != Chem.BondStereo.STEREONONE)
    return assigned, unassigned, bonds


def _fragments(smiles: str) -> list[Any] | None:
    molecules = []
    for part in (smiles or "").split("."):
        if not part:
            continue
        molecule = Chem.MolFromSmiles(part)
        if molecule is None:
            return None
        molecules.append(molecule)
    return molecules or None


def atom_deficit(product_smiles: str, precursor_smiles: str) -> dict[str, int]:
    """Elements the product has that the precursors cannot supply.

    Returns ``{element: shortfall}``; empty means the precursors carry at least as
    many of every element as the product. Surplus is ignored, because atoms lost
    going forward -- leaving groups, byproducts -- are expected.
    """

    product = Chem.MolFromSmiles(product_smiles or "")
    fragments = _fragments(precursor_smiles)
    if product is None or fragments is None:
        return {}

    product_counts = _heavy_atom_counts(product)
    precursor_counts: dict[str, int] = {}
    for fragment in fragments:
        for symbol, count in _heavy_atom_counts(fragment).items():
            precursor_counts[symbol] = precursor_counts.get(symbol, 0) + count

    return {
        symbol: count - precursor_counts.get(symbol, 0)
        for symbol, count in product_counts.items()
        if count > precursor_counts.get(symbol, 0)
    }


def atom_stereo_accounting(product_smiles: str, precursor_smiles: str) -> str:
    """Return the atom and stereo balance between a product and its precursors."""

    product = Chem.MolFromSmiles(product_smiles or "")
    if product is None:
        return "Accounting unavailable: product SMILES could not be parsed."
    fragments = _fragments(precursor_smiles)
    if fragments is None:
        return "Accounting unavailable: a precursor SMILES could not be parsed."

    product_counts = _heavy_atom_counts(product)
    precursor_counts: dict[str, int] = {}
    for fragment in fragments:
        for symbol, count in _heavy_atom_counts(fragment).items():
            precursor_counts[symbol] = precursor_counts.get(symbol, 0) + count

    rows, missing, surplus = [], [], []
    for symbol in sorted(set(product_counts) | set(precursor_counts)):
        in_product = product_counts.get(symbol, 0)
        in_precursors = precursor_counts.get(symbol, 0)
        delta = in_precursors - in_product
        rows.append(
            f"  {symbol:<3s} product={in_product:<4d} "
            f"precursors={in_precursors:<4d} delta={delta:+d}"
        )
        if delta < 0:
            missing.append(f"{-delta}x {symbol}")
        elif delta > 0:
            surplus.append(f"{delta}x {symbol}")

    product_assigned, product_unassigned, product_bonds = _stereo_counts(product)
    assigned = unassigned = bonds = 0
    for fragment in fragments:
        fragment_assigned, fragment_unassigned, fragment_bonds = _stereo_counts(fragment)
        assigned += fragment_assigned
        unassigned += fragment_unassigned
        bonds += fragment_bonds

    return "\n".join(
        [
            f"Product heavy atoms:    {sum(product_counts.values())}",
            f"Precursor heavy atoms:  {sum(precursor_counts.values())}  "
            f"(across {len(fragments)} fragment(s))",
            "Per-element:",
            *rows,
            "",
            "MISSING from precursors (present in product): "
            f"{', '.join(missing) if missing else 'none'}",
            "SURPLUS in precursors (lost in the forward step): "
            f"{', '.join(surplus) if surplus else 'none'}",
            "",
            f"Stereocentres  product: {product_assigned} assigned / "
            f"{product_unassigned} unassigned   precursors: {assigned} assigned / "
            f"{unassigned} unassigned   delta assigned={assigned - product_assigned:+d}",
            f"Stereo double bonds  product: {product_bonds}   precursors: {bonds}   "
            f"delta={bonds - product_bonds:+d}",
        ]
    )


__all__ = ["atom_deficit", "atom_stereo_accounting"]
