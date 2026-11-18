# Architecture

SynthEx is one Python distribution with a lightweight chemistry core and two optional
outer adapters.

```text
CLI / public pipeline
        |
        +--> Strategy Generator --> Route Builder --> Critic <--> Editor
        |          |                    |                  |
        |          v                    v                  v
        |    strategies.json       routes.json     reviewed_routes.json
        |                                                   |
        |                                      +------------+------------+
        |                                      |                         |
        |                                      v                         v
        |                                  Analyst              leaf completion
        |                                      |                         |
        |                                analyses.json          stitched_routes.json
        |
        +--> neutral LLMClient --> Gemini adapter
        |
        +--> ReactionJSON --> RouteJSON --> leaf completion service
                                             |
                                             +--> AiZynthFinder 4.4.1 adapter
```

`synthex.reactionjson`, `synthex.routejson`, models, and configuration parsing import
neither live adapter. Agents receive an `LLMClient` through `AgentContext`; the Editor's
function declarations are neutral `ToolSpec` objects. Only
`synthex.integrations.gemini` imports the Google SDK, and only
`synthex.integrations.aizynthfinder` imports AiZynthFinder.

The Route Builder owns the agentic strategic prefix and executes every model-proposed
ReactionJSON step before accepting it. Invalid proposals are fed back for up to three
attempts. The completion service is deliberately simpler: it canonicalizes the final
non-stock leaves, de-duplicates them, asks the pinned upstream planner for the first solved
route under the Methods budget, and grafts that fragment at each occurrence. It never asks
an LLM and never modifies the strategic prefix.

The pipeline is a state machine with immutable checkpoint payloads and one mutable
manifest. Strategies, route construction plus Critic-Editor review, analysis, and leaf
completion can be invoked independently. Resume validates the target, checkpoints, and
parent-chain digests before skipping completed stages. Each checkpoint records the exact
resolved stage configuration, allowing intentionally different models for unfinished
stages without rewriting the provenance of earlier ones. Agents never receive an output
directory and cannot write artifacts directly.
