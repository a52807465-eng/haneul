# External assets

SynthEx does not bundle licensed stocks or AiZynthFinder policy models. Before a live run,
set:

- `SYNTHEX_AIZYNTHFINDER_CONFIG` to an ordinary AiZynthFinder 4.4.1 YAML configuration
  containing the manuscript stock, expansion policy, ring-breaker policy, and filter;
- `SYNTHEX_ASSET_MANIFEST` to the benchmark-provided JSON manifest identifying those
  files and their SHA-256 digests; and
- `GEMINI_API_KEY`, or the standard Google Cloud variables when using Vertex AI.

The checked-in profile uses the Gemini API key path. For Vertex AI, set
`llm.use_vertex_ai: true` in a copied profile and provide `GOOGLE_CLOUD_PROJECT` plus
`GOOGLE_CLOUD_LOCATION` (the adapter otherwise defaults the location to `us-central1`).

The supplied `configs/synthex.manuscript.yml` records the Methods budgets. An arbitrary
local stock will not reproduce the paper results; that requires the archived asset
manifest and resolved run configuration distributed with the benchmark bundle.

AiZynthFinder separates loading entries from selecting them. The manuscript profile uses
`null` for `stocks`, `expansion_policies`, and `filter_policies`, which makes SynthEx select
every corresponding entry loaded by the AiZynthFinder YAML. Set explicit name lists in a
derived profile when only a subset should participate; an unknown name fails during
pipeline construction.
