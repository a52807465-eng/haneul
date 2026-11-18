# Installing and running the SynthEx manuscript code

This branch contains the source code for the SynthEx manuscript workflow. It also includes a
small simulated dataset and deterministic offline demo. The 1,098-target benchmark, licensed
AiZynthFinder assets, and archived result bundle are distributed separately.

## 1. System requirements

The journal reference environment is Python 3.11. Package metadata permits Python 3.11 and 3.12,
but the submitted workflow and lock file have been tested with Python 3.11.

| Component | Tested version |
|---|---|
| Operating system | macOS 26.5.1 arm64; Ubuntu 24.04 x86_64 (GitHub Actions) |
| Python | 3.11.3 on macOS; 3.11.16 on Ubuntu |
| uv | 0.6.17 on macOS; 0.12.9 on Ubuntu |
| Core dependencies | RDKit 2023.9.6, Pydantic 2.13.4, PyYAML 6.0.3 |
| Gemini adapter | Google Gen AI SDK 1.75.0 |
| Leaf completion | AiZynthFinder 4.4.1 |

Every transitive dependency and hash is recorded in `uv.lock`.

No non-standard hardware or GPU is required. The offline demo runs on a normal laptop CPU and
needs no network after installation. Allow approximately 2 GB of free disk space for the full
manuscript environment. Live planning requires internet access and a Gemini API key; leaf
completion additionally requires the separately supplied policy and stock files. Windows has not
been tested for this submission.

## 2. Installation guide

Install [uv](https://docs.astral.sh/uv/) and clone or unpack this repository. From its root, install
the core needed for the offline demo:

```bash
uv sync --python 3.11 --frozen
```

On a normal broadband-connected desktop this generally takes less than two minutes. To install
the complete manuscript environment, including Gemini and AiZynthFinder, run:

```bash
uv sync --python 3.11 --frozen --extra manuscript
```

The full installation typically takes two to five minutes, depending on network and package-cache
speed. Verify the installation without credentials or service calls:

```bash
uv run synthex reaction validate examples/reactionjson/bond-break.json
uv run synthex route validate examples/routejson/two-step-route.json
```

Expected output:

```text
valid reactionjson-1.0
valid routejson-1.0
```

## 3. Offline demo

The `demo/` directory is the required small simulated dataset:

- `input.json` defines one butane target;
- `replay-responses.json` contains five sanitized, simulated model responses; and
- `expected-summary.json` defines the result checked by the demo.

Run the complete checkpointed pipeline without credentials, network access, or external assets:

```bash
uv run python demo/run_demo.py
```

The demo executes strategy generation, ReactionJSON route construction, Critic review, analysis,
and the completion checkpoint. The Critic accepts the simulated route, so the Editor loop exits
without an edit. AiZynthFinder completion is deliberately disabled for this offline demonstration.

Expected console output (the absolute output path will differ):

```text
SynthEx offline demo: PASS
target: nature-demo (CCCC)
completed stages: strategies -> routes -> reviewed_routes -> analyses -> stitched_routes
route: 1 route with 1 step
analysis: acceptable
simulated LLM responses replayed: 5
output: .../runs/nature-demo
```

Expected runtime after installation is under 5 seconds on the tested systems and under 10 seconds
on a normal desktop. Outputs are written to `runs/nature-demo/`:

```text
manifest.json
strategies.json
routes.json
routes.tree.json
reviewed_routes.json
reviewed_routes.tree.json
analyses.json
stitched_routes.json
demo-summary.json
```

`demo-summary.json` must exactly equal `demo/expected-summary.json`; the script fails otherwise.
The manifest and checkpoint digests can be independently checked with:

```bash
uv run synthex validate-run runs/nature-demo
uv run synthex show runs/nature-demo
uv run synthex show runs/nature-demo --rendered
```

To repeat the demo, choose a new output directory because completed checkpoints are immutable:

```bash
uv run python demo/run_demo.py --output runs/nature-demo-2
```

## 4. Run SynthEx on your data

### Configure Gemini and leaf-completion assets

Copy the ignored environment template:

```bash
cp .env.example .env
```

Add the following values:

```dotenv
GEMINI_API_KEY=your-key
SYNTHEX_AIZYNTHFINDER_CONFIG=
SYNTHEX_ASSET_MANIFEST=
```

The last two values are needed only for AiZynthFinder stitching:

- `SYNTHEX_AIZYNTHFINDER_CONFIG` is an absolute path to an AiZynthFinder 4.4.1 YAML file
  loading the stock, expansion policies (including the ring-breaker policy), and filter policy.
- `SYNTHEX_ASSET_MANIFEST` is an absolute path to the companion JSON manifest containing the
  asset paths and SHA-256 digests.

Never commit `.env` or licensed assets. Validate them before making paid model calls:

```bash
uv run --env-file .env synthex config validate configs/synthex.manuscript.yml --assets
```

Omit `--assets` to validate a Gemini-only setup. The checked-in profile uses
`gemini-3.1-pro-preview`, the manuscript default. Pass `--model gemini-3.7-flash` to any LLM
command to use Flash instead. Each resolved model is recorded in the run manifest.

### Run one target

Supply a target as a valid SMILES string:

```bash
uv run --env-file .env synthex plan 'TARGET_SMILES' \
  --target-id example-1 \
  --config configs/synthex.manuscript.yml \
  --output runs/example-1
```

This runs strategy generation, route construction, Critic-Editor review, analysis, and
AiZynthFinder stitching. Runtime and Gemini cost depend on target complexity, number of strategies,
review rounds, service latency, and the number of leaves requiring completion. Resume an interrupted
run only with the same target and output directory:

```bash
uv run --env-file .env synthex plan 'TARGET_SMILES' \
  --config configs/synthex.manuscript.yml \
  --output runs/example-1 --resume
```

### Run stages separately

Each downstream stage consumes the immutable checkpoint created by the preceding stage:

```bash
# Generate three strategies with the manuscript-default Pro model.
uv run --env-file .env synthex stage strategies 'TARGET_SMILES' \
  --target-id example-1 \
  --config configs/synthex.manuscript.yml \
  --output runs/example-1

# Reuse that exact strategies.json checkpoint; use Flash downstream here.
uv run --env-file .env synthex stage routes runs/example-1 \
  --model gemini-3.7-flash --edit-rounds 2

# Analyse separately, using Pro again.
uv run --env-file .env synthex stage analyse runs/example-1 \
  --model gemini-3.1-pro-preview

# Stitch non-purchasable leaves with AiZynthFinder; this makes no LLM calls.
uv run --env-file .env synthex complete runs/example-1 \
  --config configs/synthex.manuscript.yml
```

`--edit-rounds N` is the maximum number of Editor revisions. Each unresolved round comprises one
Editor revision followed by a Critic assessment, and the loop may stop early. The manuscript
default is six; `--edit-rounds 0` skips both Critic and Editor for the route stage.

For a controlled model comparison, initialize separate runs with the same verbatim strategy:

```bash
uv run --env-file .env synthex stage strategies 'TARGET_SMILES' \
  --strategy 'FIXED_STRATEGY_TEXT' \
  --config configs/synthex.manuscript.yml \
  --output runs/fixed-pro

uv run --env-file .env synthex stage strategies 'TARGET_SMILES' \
  --strategy 'FIXED_STRATEGY_TEXT' \
  --config configs/synthex.manuscript.yml \
  --output runs/fixed-flash

uv run --env-file .env synthex stage routes runs/fixed-pro \
  --model gemini-3.1-pro-preview
uv run --env-file .env synthex stage routes runs/fixed-flash \
  --model gemini-3.7-flash
```

### Run a batch

Provide a CSV with `target_id`, `smiles`, and optional `constraint` columns:

```csv
target_id,smiles,constraint
target-1,CCCC,
target-2,CCO,"Prefer a convergent route"
```

```bash
uv run --env-file .env synthex batch targets.csv \
  --config configs/synthex.manuscript.yml \
  --output runs/batch
```

## 5. Outputs and verification

A live run creates `manifest.json` and, as stages complete, `strategies.json`, `routes.json`,
`reviewed_routes.json`, `analyses.json`, and `stitched_routes.json`. Inspect and verify them with:

```bash
uv run synthex validate-run runs/example-1
uv run synthex show runs/example-1
uv run synthex show runs/example-1 --rendered
uv run synthex show runs/example-1 --format tree
```

The manifest records the resolved per-stage configuration, model, prompt and code identifiers,
external-asset digests, timings, and checkpoint hashes. It never stores the API key. ReactionJSON
and RouteJSON are documented in `docs/reactionjson.md` and `docs/routejson.md`.

## 6. Reproducing the manuscript results

The code environment is fixed by `uv.lock`, and the Methods parameters are in
`configs/synthex.manuscript.yml`. Live Gemini generation is stochastic because the manuscript did
not use a sampling seed; reruns are therefore expected to be qualitatively rather than
byte-for-byte identical.

Quantitative reproduction additionally requires the separately supplied 1,098-target benchmark,
resolved run configuration, Gemini access, and exact AiZynthFinder stock/policy asset manifest.
Those materials own target selection, the template-only baseline, aggregation, and result tables.
After installing this repository, the companion benchmark can call the public `synthex batch`
command and consume the versioned run/checkpoint schemas in `schemas/`.
