# External benchmark interface 1.0

The separately shipped benchmark supplies target records with `target_id`, `smiles`, and
an optional `constraint`. It invokes `synthex batch` or the public `SynthexPipeline`, then
reads only `manifest.json` and its digest-linked checkpoint files.

SynthEx owns per-target strategies, routes, Critic/Editor lineage, analysis, stitched
routes, timing, and structured failures. The benchmark owns the 1,098-target collection,
template-only baseline, aggregation, reported solve rates, and manuscript figures.

The machine-readable contract is `schemas/benchmark-interface-1.0.schema.json`.
