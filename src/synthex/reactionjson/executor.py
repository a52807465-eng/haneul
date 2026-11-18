"""Deterministic executor for LLM-proposed atomic-op sequences.

Resolves map-number arguments to atom indices at runtime and applies each op
to the molecule state left by the previous op.
"""

from __future__ import annotations

from rdkit import Chem

from synthex.reactionjson.operations import (
    add_bond,
    add_group,
    break_bond,
    change_atom,
    change_bond_order,
    clear_stereocenter,
    invert_stereocenter,
    remove_group,
    set_bond_stereo,
    set_explicit_h,
)


def get_mapped_smiles(smiles: str) -> str:
    """Return SMILES where every atom carries map number = atom_idx + 1."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return Chem.MolToSmiles(mol)


def _resolve_map(mol: Chem.Mol, map_num: int) -> int:
    """Return the atom index for a given map number.

    Raises ValueError if no atom carries the map number, or if more than one
    does. The ambiguity check is important: atoms that ``add_group`` introduces
    without an explicit map number all share map 0, so a reference to map 0 (or
    any duplicated map number) would otherwise silently resolve to whichever
    atom happens to come first and corrupt the molecule.
    """
    matches = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() == map_num]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        surviving = sorted({a.GetAtomMapNum() for a in mol.GetAtoms() if a.GetAtomMapNum() > 0})
        raise ValueError(
            f"No atom with map number {map_num} in current molecule "
            f"(heavy atom count: {mol.GetNumAtoms()}). "
            f"Surviving map numbers: {surviving}"
        )
    raise ValueError(
        f"Ambiguous map number {map_num}: {len(matches)} atoms carry it. "
        f"Atoms introduced by add_group without an explicit map number all "
        f"share map 0; tag any atom you need to reference later with a unique, "
        f"unused map number in the fragment SMILES."
    )


def execute_ops(
    ops: list[dict],
    mapped_smiles: str,
) -> tuple[str | None, str | None]:
    """Apply a sequence of atomic ops to the molecule encoded in mapped_smiles.

    Returns (result_smiles, error_message). Exactly one will be None.
    Map numbers in each op dict are resolved to atom indices at the time that
    op executes, so they remain valid even after prior ops remove or add atoms.
    Atom map numbers are always preserved in the returned SMILES. Atoms
    introduced by add_group come out with map=0 *unless* the fragment SMILES
    gives them an explicit map number, which is preserved so that later ops can
    reference the introduced atom (e.g. to close a ring onto it).
    """
    mol = Chem.MolFromSmiles(mapped_smiles)
    if mol is None:
        return None, f"Could not parse mapped SMILES: {mapped_smiles!r}"

    for i, op_dict in enumerate(ops):
        op = op_dict.get("op")
        try:
            if op == "break_bond":
                mol = break_bond(
                    mol, _resolve_map(mol, op_dict["map_a"]), _resolve_map(mol, op_dict["map_b"])
                )
            elif op == "add_bond":
                mol = add_bond(
                    mol,
                    _resolve_map(mol, op_dict["map_a"]),
                    _resolve_map(mol, op_dict["map_b"]),
                    op_dict.get("order", 1),
                )
            elif op == "change_bond_order":
                mol = change_bond_order(
                    mol,
                    _resolve_map(mol, op_dict["map_a"]),
                    _resolve_map(mol, op_dict["map_b"]),
                    op_dict["delta"],
                )
            elif op == "change_atom":
                mol = change_atom(mol, _resolve_map(mol, op_dict["map_idx"]), op_dict["symbol"])
            elif op == "set_explicit_h":
                mol = set_explicit_h(mol, _resolve_map(mol, op_dict["map_idx"]), op_dict["n"])
            elif op == "add_group":
                mol = add_group(
                    mol,
                    _resolve_map(mol, op_dict["map_idx"]),
                    op_dict["fragment_smiles"],
                    op_dict.get("order", 1),
                )
            elif op == "remove_group":
                mol = remove_group(
                    mol,
                    _resolve_map(mol, op_dict["map_idx"]),
                    _resolve_map(mol, op_dict["map_root_idx"]),
                )
            elif op == "invert_stereocenter":
                mol = invert_stereocenter(mol, _resolve_map(mol, op_dict["map_idx"]))
            elif op == "clear_stereocenter":
                mol = clear_stereocenter(mol, _resolve_map(mol, op_dict["map_idx"]))
            elif op == "set_bond_stereo":
                ref_a = (
                    _resolve_map(mol, op_dict["map_ref_a"]) if op_dict.get("map_ref_a") else None
                )
                ref_b = (
                    _resolve_map(mol, op_dict["map_ref_b"]) if op_dict.get("map_ref_b") else None
                )
                mol = set_bond_stereo(
                    mol,
                    _resolve_map(mol, op_dict["map_a"]),
                    _resolve_map(mol, op_dict["map_b"]),
                    op_dict["stereo"],
                    ref_a,
                    ref_b,
                )
            else:
                return None, f"Op [{i}]: unknown operation {op!r}"
        except (
            ValueError,
            KeyError,
            RuntimeError,
            Chem.AtomKekulizeException,
            Chem.KekulizeException,
            Chem.AtomValenceException,
        ) as exc:
            return None, f"Op [{i}] {op}: {type(exc).__name__}: {exc}\n  op_dict: {op_dict}"

    try:
        rw = Chem.RWMol(mol)
        clean = rw.GetMol()
        Chem.AssignStereochemistry(clean, cleanIt=True, force=True)
        result = Chem.MolToSmiles(clean)
    except (
        ValueError,
        RuntimeError,
        Chem.AtomKekulizeException,
        Chem.KekulizeException,
        Chem.AtomValenceException,
    ) as exc:
        return None, f"Finalize: {type(exc).__name__}: {exc}"
    if not result:
        return None, "Ops produced an empty molecule"
    return result, None
