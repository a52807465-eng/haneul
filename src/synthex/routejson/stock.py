"""Lightweight stock-file → stock_check helper.

The route_ops render path takes an optional ``stock_check`` callable
``(canonical_smiles) -> bool``. Building one against the full search
``Configuration`` pulls in template engine, ONNX filter, etc. — wasteful
when all we want is in-stock lookup.

This module loads the InChI-key column of an eMolecule-style CSV (or a
plain text file with one InChI key per line, or an HDF5 with a ``table``
dataset) into a Python set and returns a closure that runs:

    InChIKey(canonical SMILES) -> str  →  membership in the in-memory set.

Same lookup the canonical ``InMemoryInchiKeyQuery`` does, minus all the
configuration plumbing. Membership is exact: there is no size-based shortcut, so a
molecule is in stock only when its InChI key is in the loaded set.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import inchi


def load_stock_check(
    path: str | Path,
    inchi_key_col: str = "inchi_key",
) -> Callable[[str], bool]:
    """Return a ``stock_check(smiles) -> bool`` closure backed by ``path``.

    Parameters
    ----------
    path
        File holding the stock's InChI keys. Supported formats:
          * ``.csv`` — pandas-readable; ``inchi_key_col`` selects the column.
          * ``.h5`` / ``.hdf5`` — pandas HDF5 with a ``table`` dataset.
          * any other extension — treated as a plain text file, one
            InChI key per line.
    inchi_key_col
        Column name when ``path`` is a CSV / HDF5 dataframe.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"stock file not found: {p}")

    keys = _load_inchi_keys(p, inchi_key_col)

    def stock_check(smiles: str) -> bool:
        if not smiles:
            return False
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            key = inchi.MolToInchiKey(mol)
        except Exception:
            return False
        return key in keys

    # Stash the set on the closure so callers can inspect / reuse / sample.
    stock_check.inchi_keys = keys  # type: ignore[attr-defined]
    return stock_check


def _load_inchi_keys(p: Path, inchi_key_col: str) -> set[str]:
    """Load InChI keys from ``p`` into a set. Format inferred from suffix."""
    ext = p.suffix.lower()
    if ext == ".csv":
        # pandas is the only fast option for 23M-row CSVs; only load the
        # one column we need.
        import pandas as pd

        df = pd.read_csv(p, usecols=[inchi_key_col])
        return set(df[inchi_key_col].dropna().values)
    if ext in (".h5", ".hdf5"):
        import pandas as pd

        df = pd.read_hdf(p, key="table")
        return set(df[inchi_key_col].dropna().values)
    # Plain text: one key per line.
    with open(p) as f:
        return {line.strip() for line in f if line.strip()}
