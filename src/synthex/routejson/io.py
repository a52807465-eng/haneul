"""Save / load route-ops plans as YAML (default) or JSON.

YAML output uses inline (flow) style for short dicts (ops, frontier
entries) and block style for the step list, so the per-step structure
stays scannable while each op fits on a single line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

_FLOW_KEYS = {"op", "smi", "in_stock", "smiles"}  # dicts with any of these keys → flow style


def _represent_dict_flow_if_small(dumper: yaml.SafeDumper, data: dict) -> yaml.MappingNode:
    """Use flow style for tiny dicts (ops, frontier entries); block otherwise."""
    flow = bool(data) and any(k in _FLOW_KEYS for k in data.keys())
    return dumper.represent_mapping("tag:yaml.org,2002:map", data, flow_style=flow)


class _PlanDumper(yaml.SafeDumper):
    """SafeDumper w/ a per-dict flow heuristic for plan files."""


_PlanDumper.add_representer(dict, _represent_dict_flow_if_small)


def plan_to_yaml(plan: dict[str, Any]) -> str:
    """Serialise a plan dict to a YAML string."""
    return yaml.dump(
        plan,
        Dumper=_PlanDumper,
        sort_keys=False,
        allow_unicode=True,
        width=200,
        default_flow_style=False,
    )


def plan_from_yaml(yaml_str: str) -> dict[str, Any]:
    """Parse a YAML plan string back into a dict."""
    return yaml.safe_load(yaml_str)


def save_plan(plan: dict[str, Any], path: str | Path) -> None:
    """Write a plan to disk. YAML by default; .json extension switches format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".json":
        p.write_text(json.dumps(plan, indent=2))
    else:
        p.write_text(plan_to_yaml(plan))


def load_plan(path: str | Path) -> dict[str, Any]:
    """Read a plan from disk; format inferred from extension."""
    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() == ".json":
        return json.loads(text)
    return plan_from_yaml(text)
