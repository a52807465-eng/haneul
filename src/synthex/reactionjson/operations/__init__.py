"""Atomic molecular-editing operations for LLM-composed retrosynthesis.

Each op takes an RDKit Mol plus atom-level arguments, returns a new sanitized
Mol, and raises ValueError on any invalid input or invalid resulting molecule.
Input Mols are never mutated.
"""

from .add_bond import add_bond
from .add_group import add_group
from .break_bond import break_bond
from .change_atom import change_atom
from .change_bond_order import change_bond_order
from .clear_stereocenter import clear_stereocenter
from .invert_stereocenter import invert_stereocenter
from .remove_group import remove_group
from .set_bond_stereo import set_bond_stereo
from .set_explicit_h import set_explicit_h

__all__ = [
    "add_bond",
    "break_bond",
    "change_atom",
    "set_explicit_h",
    "invert_stereocenter",
    "clear_stereocenter",
    "change_bond_order",
    "set_bond_stereo",
    "add_group",
    "remove_group",
]
