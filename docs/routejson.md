# RouteJSON 1.0

RouteJSON is an editable, atom-map-anchored sequence of ReactionJSON steps. Its target is
fully mapped and step identifiers are unique `S1`, `S2`, … values. Each step's operation
list acts on exactly one current frontier molecule. Rendering re-executes the operations,
selects that molecule from the live frontier by map identity, and derives reaction SMILES,
precursor frontiers, leaf flags, and stock status without an LLM call.

The lean RouteJSON document stores authored content only: target, operations, description,
rationale, conditions, key/supporting label, and optional metadata. Derived `rendered` and
`route` fields are not persisted in the authoritative document.

The authoritative machine schema is `schemas/routejson-1.0.schema.json`; executable
documents are under `examples/routejson/`.
