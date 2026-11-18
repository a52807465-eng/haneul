"""Command-line interface for the SynthEx manuscript artifact."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class CliError(ValueError):
    """An expected command-line input error."""


def _read_json(path: str | Path) -> Any:
    source = Path(path)
    try:
        return json.loads(source.read_text())
    except OSError as exc:
        raise CliError(f"cannot read {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CliError(f"invalid JSON in {source}: {exc}") from exc


def _write_text(text: str, output: str | None) -> None:
    if output:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
    else:
        print(text)


def _reaction_validate(args: argparse.Namespace) -> None:
    from synthex.reactionjson import Reaction

    reaction = Reaction.model_validate(_read_json(args.path))
    print(f"valid {reaction.format_version}")


def _reaction_apply(args: argparse.Namespace) -> None:
    from synthex.reactionjson import Reaction, apply_reaction

    if args.path:
        reaction = Reaction.model_validate(_read_json(args.path))
    else:
        if not args.product or not args.operations:
            raise CliError("provide a ReactionJSON file or both --product and --operations")
        operations = _read_json(args.operations)
        if not isinstance(operations, list):
            raise CliError("operations JSON must contain a list")
        reaction = Reaction(product=args.product, operations=operations)
    print(apply_reaction(reaction))


def _route_validate(args: argparse.Namespace) -> None:
    from synthex.routejson import Route

    route = Route.model_validate(_read_json(args.path))
    print(f"valid {route.format_version}")


def _route_render(args: argparse.Namespace) -> None:
    from synthex.routejson import Route, plan_to_yaml, render_plan

    route = Route.model_validate(_read_json(args.path))
    rendered = render_plan(route.to_op_plan(), strip_frontier_maps=args.strip_maps)
    if args.output and Path(args.output).suffix.lower() == ".json":
        text = json.dumps(rendered, indent=2) + "\n"
    else:
        text = plan_to_yaml(rendered)
    _write_text(text, args.output)


def _schema_export(args: argparse.Namespace) -> None:
    from synthex.artifacts import benchmark_interface_schema, run_manifest_schema
    from synthex.reactionjson import reaction_schema
    from synthex.routejson import route_schema

    schemas = {
        "benchmark-interface-1.0.schema.json": benchmark_interface_schema(),
        "reactionjson-1.0.schema.json": reaction_schema(),
        "routejson-1.0.schema.json": route_schema(),
        "run-manifest-1.0.schema.json": run_manifest_schema(),
    }
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, schema in schemas.items():
        (destination / filename).write_text(json.dumps(schema, indent=2) + "\n")
    print(f"wrote {len(schemas)} schemas to {destination}")


def _load_live_pipeline(
    config_path: str | None,
    *,
    run_dir: str | Path | None = None,
    model: str | None = None,
    strategy_count: int | None = None,
    route_steps: int | None = None,
    edit_rounds: int | None = None,
    include_completion: bool = True,
):
    from synthex.artifacts import ArtifactStore
    from synthex.config import ManuscriptConfig, apply_runtime_overrides, load_config

    if config_path is not None:
        config = load_config(config_path)
    elif run_dir is not None:
        manifest = ArtifactStore(run_dir).validate()
        config = ManuscriptConfig.model_validate(manifest.config)
    else:
        raise CliError("a configuration is required when creating a run")
    config = apply_runtime_overrides(
        config,
        model=model,
        strategy_count=strategy_count,
        route_step_limit=route_steps,
        edit_iteration_limit=edit_rounds,
    )

    try:
        from synthex.application import create_pipeline
    except ModuleNotFoundError as exc:
        if exc.name in {"aizynthfinder", "google", "google.genai"}:
            raise CliError(
                "the live pipeline requires `pip install 'synthex[manuscript]'`"
            ) from exc
        raise
    return create_pipeline(config, include_completion=include_completion)


def _plan(args: argparse.Namespace) -> None:
    pipeline = _load_live_pipeline(
        args.config,
        model=args.model,
        strategy_count=args.strategy_count,
        route_steps=args.route_steps,
        edit_rounds=args.edit_rounds,
    )
    manifest = asyncio.run(
        pipeline.run(
            args.target,
            args.output,
            target_id=args.target_id,
            constraint=args.constraint,
            resume=args.resume,
        )
    )
    print(f"{manifest.status} run {manifest.run_id}: {Path(args.output) / 'manifest.json'}")


def _safe_target_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not safe:
        raise CliError(f"target id {value!r} cannot be used as a directory name")
    return safe


def _batch(args: argparse.Namespace) -> None:
    from synthex.artifacts import BenchmarkResult

    pipeline = _load_live_pipeline(
        args.config,
        model=args.model,
        strategy_count=args.strategy_count,
        route_steps=args.route_steps,
        edit_rounds=args.edit_rounds,
    )
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    with Path(args.targets).open(newline="") as stream:
        records = list(csv.DictReader(stream))

    async def run_all() -> list[dict[str, Any]]:
        batch_results: list[dict[str, Any]] = []
        for index, record in enumerate(records, start=1):
            target_id = record.get("target_id") or record.get("id") or f"target-{index}"
            smiles = record.get("smiles") or record.get("target")
            if not smiles:
                raise CliError(f"CSV row {index} has no smiles/target value")
            run_dir = root / _safe_target_id(target_id)
            try:
                await pipeline.run(
                    smiles,
                    run_dir,
                    target_id=target_id,
                    constraint=record.get("constraint") or None,
                    resume=args.resume,
                )
                result = BenchmarkResult(
                    target_id=target_id,
                    status="complete",
                    run_manifest=str(Path(run_dir.name) / "manifest.json"),
                    strategic_routes=str(Path(run_dir.name) / "reviewed_routes.json"),
                    stitched_routes=str(Path(run_dir.name) / "stitched_routes.json"),
                )
            except Exception as exc:
                result = BenchmarkResult(
                    target_id=target_id,
                    status="failed",
                    run_manifest=str(Path(run_dir.name) / "manifest.json"),
                    failure_category=type(exc).__name__,
                )
            batch_results.append(result.model_dump(mode="json"))
        return batch_results

    results = asyncio.run(run_all())
    (root / "benchmark-results.json").write_text(json.dumps(results, indent=2) + "\n")
    failures = sum(item["status"] == "failed" for item in results)
    print(f"processed {len(results)} targets ({failures} failed)")
    if failures:
        raise CliError(f"{failures} batch target(s) failed; inspect benchmark-results.json")


def _resume_stages(
    args: argparse.Namespace,
    stages: tuple[str, ...],
    *,
    include_completion: bool,
) -> None:
    from synthex.artifacts import ArtifactStore

    store = ArtifactStore(args.run_dir)
    manifest = store.validate()
    pipeline = _load_live_pipeline(
        getattr(args, "config", None),
        run_dir=args.run_dir,
        model=getattr(args, "model", None),
        route_steps=getattr(args, "route_steps", None),
        edit_rounds=getattr(args, "edit_rounds", None),
        include_completion=include_completion,
    )
    resumed = asyncio.run(
        pipeline.run(
            manifest.target,
            args.run_dir,
            target_id=manifest.target_id,
            constraint=manifest.constraint,
            resume=True,
            stages=stages,
        )
    )
    print(f"{resumed.status} run {resumed.run_id}")


def _stage_strategies(args: argparse.Namespace) -> None:
    if args.strategy and args.strategy_count is not None:
        raise CliError("--strategy and --strategy-count are mutually exclusive")
    pipeline = _load_live_pipeline(
        args.config,
        model=args.model,
        strategy_count=args.strategy_count,
        include_completion=False,
    )
    manifest = asyncio.run(
        pipeline.run(
            args.target,
            args.output,
            target_id=args.target_id,
            constraint=args.constraint,
            resume=args.resume,
            stages=("strategies",),
            seed_strategies=args.strategy,
        )
    )
    print(f"{manifest.status} run {manifest.run_id}: {Path(args.output) / 'strategies.json'}")


def _stage_routes(args: argparse.Namespace) -> None:
    _resume_stages(
        args,
        ("routes", "reviewed_routes"),
        include_completion=False,
    )


def _stage_analyse(args: argparse.Namespace) -> None:
    _resume_stages(args, ("analyses",), include_completion=False)


def _complete(args: argparse.Namespace) -> None:
    _resume_stages(args, ("stitched_routes",), include_completion=True)


def _show(args: argparse.Namespace) -> None:
    """Render a run's routes as lean plan YAML instead of checkpoint JSON.

    A read-only view. Checkpoints are digest-linked and immutable, so this
    reformats on the way out and never rewrites them.
    """
    from synthex.artifacts import ArtifactStore
    from synthex.routejson import Route, op_plan_to_strategy_tree, plan_to_yaml, render_plan

    store = ArtifactStore(args.run_dir)
    manifest = store.validate()
    stage = args.stage
    if stage is None:
        for candidate in ("reviewed_routes", "routes"):
            if candidate in manifest.artifacts:
                stage = candidate
                break
        else:
            raise CliError(
                f"{store.root} has no routes yet (completed: "
                f"{', '.join(manifest.completed_stages) or 'none'})"
            )
    if stage not in manifest.artifacts:
        raise CliError(
            f"stage {stage!r} is not complete in {store.root}; completed: "
            f"{', '.join(manifest.completed_stages) or 'none'}"
        )

    payload = store.read_checkpoint(manifest.artifacts[stage])
    if args.index is not None:
        if not 0 <= args.index < len(payload):
            raise CliError(f"--index {args.index} out of range [0, {len(payload) - 1}]")
        payload = [payload[args.index]]

    if args.format == "tree":
        trees = []
        for item in payload:
            if "route" not in item:
                # Per-item stage isolation deliberately preserves failed
                # strategies in the checkpoint. Keep that diagnostic in the
                # view rather than making one failure hide every good route.
                trees.append(item)
                continue
            tree = op_plan_to_strategy_tree(Route.model_validate(item["route"]).to_op_plan())
            if item.get("status") not in {None, "ok"}:
                tree = {
                    "status": item["status"],
                    **({"failure": item["failure"]} if item.get("failure") else {}),
                } | tree
            trees.append(tree)
        document: Any = trees[0] if args.index is not None else trees
        _write_text(json.dumps(document, indent=2) + "\n", args.output)
        return

    blocks: list[str] = []
    for offset, item in enumerate(payload):
        position = args.index if args.index is not None else offset
        header = [f"# {stage}[{position}] of {manifest.target_id or manifest.run_id}"]
        strategy = item.get("strategy")
        if strategy:
            header.append(f"# strategy: {strategy}")
        status = item.get("status")
        if status not in {None, "ok"}:
            header.append(f"# status: {status}")
        failure = item.get("failure")
        if failure:
            header.append(
                f"# failure: {failure.get('category', 'unknown')}: {failure.get('message', '')}"
            )
        if "route" not in item:
            blocks.append("\n".join(header) + "\n" + plan_to_yaml(item).rstrip())
            continue
        plan = Route.model_validate(item["route"]).to_op_plan()
        if args.rendered:
            plan = render_plan(plan, strip_frontier_maps=args.strip_maps)
        critique = item.get("final_critique")
        if critique is not None:
            verdict = "satisfactory" if critique.get("is_satisfactory") else "NOT satisfactory"
            header.append(f"# critic: {verdict}")
            if critique.get("overall_feedback"):
                header.append(f"#   {critique['overall_feedback']}")
            for finding in critique.get("problematic_steps") or []:
                header.append(
                    f"#   [{finding.get('severity')}] {finding.get('step_id')}: "
                    f"{finding.get('feedback')}"
                )
        blocks.append("\n".join(header) + "\n" + plan_to_yaml(plan).rstrip())

    _write_text("---\n".join(block + "\n" for block in blocks), args.output)


def _validate_run(args: argparse.Namespace) -> None:
    from synthex.artifacts import ArtifactStore

    manifest = ArtifactStore(args.run_dir).validate()
    print(
        f"valid {manifest.format_version}: {manifest.status}, "
        f"{len(manifest.artifacts)} checkpoint(s)"
    )


def _import_legacy(args: argparse.Namespace) -> None:
    from synthex.routejson import Route, load_plan

    source = Path(args.legacy_run)
    candidates = sorted(source.glob("plan*.yaml")) + sorted(source.glob("plan*.json"))
    if not candidates:
        raise CliError(f"no plan*.yaml or plan*.json found in {source}")
    selected = candidates[-1]
    route = Route.from_op_plan(load_plan(selected))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": "legacy-import-1.0",
        "source_filename": selected.name,
        "route": route.model_dump(mode="json"),
    }
    (output / "route.imported.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"imported {selected.name} to {output / 'route.imported.json'}")


def _config_validate(args: argparse.Namespace) -> None:
    from synthex.config import load_config, validate_external_assets

    config = load_config(args.path)
    paths = validate_external_assets(config) if args.assets else []
    print(
        f"valid {config.format_version}"
        + (f"; {len(paths)} asset path(s) found" if args.assets else "")
    )


def _add_model_override(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        help=(
            "Gemini model override for the stages run by this command "
            "(for example gemini-3.1-pro-preview or gemini-3.7-flash)"
        ),
    )


def _add_edit_round_override(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--edit-rounds",
        type=int,
        help=("maximum Critic-Editor improvement rounds; 0 skips both agents for this route stage"),
    )


def _add_strategy_count_override(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--strategy-count",
        type=int,
        help="number of independent strategies (manuscript default: 3)",
    )


def _add_route_step_override(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--route-steps",
        type=int,
        help="maximum strategic ReactionJSON steps per route (manuscript default: 25)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synthex", description="Strategy-first synthesis planning"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    reaction = subcommands.add_parser("reaction", help="ReactionJSON tools")
    reaction_commands = reaction.add_subparsers(dest="reaction_command", required=True)
    reaction_validate = reaction_commands.add_parser("validate", help="validate a document")
    reaction_validate.add_argument("path")
    reaction_validate.set_defaults(handler=_reaction_validate)
    reaction_apply = reaction_commands.add_parser("apply", help="execute graph edits")
    reaction_apply.add_argument("path", nargs="?", help="ReactionJSON document")
    reaction_apply.add_argument("--product", help="atom-mapped product SMILES")
    reaction_apply.add_argument("--operations", help="JSON file containing an operation list")
    reaction_apply.set_defaults(handler=_reaction_apply)

    route = subcommands.add_parser("route", help="RouteJSON tools")
    route_commands = route.add_subparsers(dest="route_command", required=True)
    route_validate = route_commands.add_parser("validate", help="validate a document")
    route_validate.add_argument("path")
    route_validate.set_defaults(handler=_route_validate)
    route_render = route_commands.add_parser("render", help="execute and render a route")
    route_render.add_argument("path")
    route_render.add_argument("--output", "-o")
    route_render.add_argument("--strip-maps", action="store_true")
    route_render.set_defaults(handler=_route_render)

    schema = subcommands.add_parser("schema", help="JSON Schema tools")
    schema_commands = schema.add_subparsers(dest="schema_command", required=True)
    schema_export = schema_commands.add_parser("export", help="export authoritative schemas")
    schema_export.add_argument("--output", "-o", default="schemas")
    schema_export.set_defaults(handler=_schema_export)

    config = subcommands.add_parser("config", help="configuration tools")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_validate = config_commands.add_parser("validate", help="validate configuration")
    config_validate.add_argument("path")
    config_validate.add_argument(
        "--assets", action="store_true", help="also require external files"
    )
    config_validate.set_defaults(handler=_config_validate)

    plan = subcommands.add_parser("plan", help="run all manuscript stages for one target")
    plan.add_argument("target")
    plan.add_argument("--config", required=True)
    plan.add_argument("--output", required=True)
    plan.add_argument("--target-id")
    plan.add_argument("--constraint")
    plan.add_argument("--resume", action="store_true")
    _add_model_override(plan)
    _add_strategy_count_override(plan)
    _add_route_step_override(plan)
    _add_edit_round_override(plan)
    plan.set_defaults(handler=_plan)

    batch = subcommands.add_parser("batch", help="run targets from a CSV file")
    batch.add_argument("targets")
    batch.add_argument("--config", required=True)
    batch.add_argument("--output", required=True)
    batch.add_argument("--resume", action="store_true")
    _add_model_override(batch)
    _add_strategy_count_override(batch)
    _add_route_step_override(batch)
    _add_edit_round_override(batch)
    batch.set_defaults(handler=_batch)

    stage = subcommands.add_parser("stage", help="run manuscript stages independently")
    stage_commands = stage.add_subparsers(dest="stage_command", required=True)

    stage_strategies = stage_commands.add_parser(
        "strategies", help="generate and checkpoint strategies only"
    )
    stage_strategies.add_argument("target")
    stage_strategies.add_argument("--config", required=True)
    stage_strategies.add_argument("--output", required=True)
    stage_strategies.add_argument("--target-id")
    stage_strategies.add_argument("--constraint")
    stage_strategies.add_argument(
        "--strategy",
        action="append",
        help=(
            "provided strategy text; repeat for multiple strategies and bypass "
            "the Strategy Generator"
        ),
    )
    stage_strategies.add_argument("--resume", action="store_true")
    _add_model_override(stage_strategies)
    _add_strategy_count_override(stage_strategies)
    stage_strategies.set_defaults(handler=_stage_strategies)

    stage_routes = stage_commands.add_parser(
        "routes", help="build routes and run the Critic-Editor loop"
    )
    stage_routes.add_argument("run_dir")
    stage_routes.add_argument(
        "--config", help="optional base config; defaults to the run creation config"
    )
    _add_model_override(stage_routes)
    _add_route_step_override(stage_routes)
    _add_edit_round_override(stage_routes)
    stage_routes.set_defaults(handler=_stage_routes)

    stage_analyse = stage_commands.add_parser("analyse", help="analyse previously reviewed routes")
    stage_analyse.add_argument("run_dir")
    stage_analyse.add_argument(
        "--config", help="optional base config; defaults to the run creation config"
    )
    _add_model_override(stage_analyse)
    stage_analyse.set_defaults(handler=_stage_analyse)

    complete = subcommands.add_parser("complete", help="resume a run through leaf completion")
    complete.add_argument("run_dir")
    complete.add_argument(
        "--config", help="optional config with completion assets; defaults to run config"
    )
    complete.set_defaults(handler=_complete)

    show = subcommands.add_parser("show", help="print a run's routes as readable plan YAML")
    show.add_argument("run_dir")
    show.add_argument(
        "--stage",
        choices=["routes", "reviewed_routes"],
        help="which checkpoint to read; defaults to reviewed_routes when present",
    )
    show.add_argument("--index", type=int, help="show only this route (0-based)")
    show.add_argument(
        "--format",
        choices=["plan", "tree"],
        default="plan",
        help=(
            "plan: lean route-ops YAML (default). tree: molecule-centric strategy "
            "tree JSON, one node per molecule with its reaction inlined"
        ),
    )
    show.add_argument(
        "--rendered",
        action="store_true",
        help="also execute the ops and include each step's frontier",
    )
    show.add_argument(
        "--strip-maps", action="store_true", help="with --rendered, unmap frontier SMILES"
    )
    show.add_argument("--output", "-o")
    show.set_defaults(handler=_show)

    validate_run = subcommands.add_parser("validate-run", help="verify manifest/checkpoint digests")
    validate_run.add_argument("run_dir")
    validate_run.set_defaults(handler=_validate_run)

    import_legacy = subcommands.add_parser("import-legacy", help="import a historical plan")
    import_legacy.add_argument("legacy_run")
    import_legacy.add_argument("--output", required=True)
    import_legacy.set_defaults(handler=_import_legacy)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except Exception as exc:
        from synthex.artifacts import redact_sensitive_text

        print(f"synthex: error: {redact_sensitive_text(str(exc))}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
