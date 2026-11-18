# ReactionJSON 1.0

ReactionJSON is SynthEx's versioned, deterministic representation of one retrosynthetic
disconnection. The `product` is a parseable SMILES in which every atom has one unique,
positive map number. `operations` execute in list order against the molecule left by the
previous operation and produce one or more dot-separated, atom-mapped precursors.

The direction is always:

```text
mapped product --ReactionJSON operations--> mapped precursors
```

The language is closed: version 1.0 contains exactly ten operations.

| Operation | Required fields | Optional fields |
|---|---|---|
| `break_bond` | `map_a`, `map_b` | — |
| `add_bond` | `map_a`, `map_b` | `order` (1, 2, or 3; default 1) |
| `change_bond_order` | `map_a`, `map_b`, `delta` (-2, -1, 1, or 2) | — |
| `change_atom` | `map_idx`, `symbol` | — |
| `set_explicit_h` | `map_idx`, `n` | — |
| `add_group` | `map_idx`, `fragment_smiles` | `order` (default 1) |
| `remove_group` | `map_idx`, `map_root_idx` | — |
| `invert_stereocenter` | `map_idx` | — |
| `clear_stereocenter` | `map_idx` | — |
| `set_bond_stereo` | `map_a`, `map_b`, `stereo` (`E` or `Z`) | `map_ref_a`, `map_ref_b` |

All atom references are map numbers, never transient RDKit atom indices. Product-derived
atoms preserve their maps. `add_group` may introduce an explicitly mapped fragment atom
so a later operation in the same list can reference it; otherwise new atoms initially
have map zero and receive collision-free maps when a RouteJSON frontier is materialized.

Operations reject missing/ambiguous maps, impossible valence, invalid bonds, and invalid
stereochemistry. `break_bond` is for sigma bonds; use `change_bond_order` for pi-bond
changes. Bond removal clears affected tetrahedral tags, and callers must explicitly
restore intended stereochemistry.

The authoritative machine schema is `schemas/reactionjson-1.0.schema.json`; executable
documents are under `examples/reactionjson/`.
