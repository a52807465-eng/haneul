"""Strict, side-effect-free configuration for the manuscript pipeline."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from synthex.completion import CompletionBudget

_ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMConfig(ConfigModel):
    provider: Literal["gemini"] = "gemini"
    model: str = "gemini-3.1-pro-preview"
    use_vertex_ai: bool = False
    timeout_seconds: float = Field(default=600, gt=0)
    max_attempts: int = Field(default=3, ge=1)
    thinking_level: Literal["minimal", "low", "medium", "high"] | None = None


class AgentConfig(ConfigModel):
    strategy_count: int = Field(default=3, ge=1)
    route_step_limit: int = Field(default=25, ge=1)
    route_proposal_attempts: int = Field(default=3, ge=1)
    edit_iteration_limit: int = Field(default=6, ge=0)
    editor_turn_limit: int = Field(default=60, ge=1)
    strategy_temperature: float = Field(default=0.1, ge=0, le=2)
    route_temperature: float = Field(default=0.3, ge=0, le=2)
    critic_temperature: float = Field(default=0.1, ge=0, le=2)
    editor_temperature: float = Field(default=0.3, ge=0, le=2)
    analyst_temperature: float = Field(default=0.3, ge=0, le=2)
    strategy_thinking_level: Literal["minimal", "low", "medium", "high"] | None = None
    route_thinking_level: Literal["minimal", "low", "medium", "high"] | None = "medium"
    critic_thinking_level: Literal["minimal", "low", "medium", "high"] | None = "medium"
    editor_thinking_level: Literal["minimal", "low", "medium", "high"] | None = "medium"
    analyst_thinking_level: Literal["minimal", "low", "medium", "high"] | None = None


class CompletionConfig(ConfigModel):
    enabled: bool = True
    aizynthfinder_config: Path | None = None
    stocks: list[str] | None = None
    expansion_policies: list[str] | None = None
    filter_policies: list[str] | None = None
    budget: CompletionBudget = Field(default_factory=CompletionBudget)


class AssetEntry(ConfigModel):
    identifier: str
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ExternalAssetManifest(ConfigModel):
    format_version: Literal["external-assets-1.0"] = "external-assets-1.0"
    assets: list[AssetEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_identifiers(self) -> ExternalAssetManifest:
        identifiers = [asset.identifier for asset in self.assets]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("external asset identifiers must be unique")
        return self


class ManuscriptConfig(ConfigModel):
    format_version: Literal["synthex-config-1.0"] = "synthex-config-1.0"
    profile: Literal["manuscript"] = "manuscript"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    completion: CompletionConfig = Field(default_factory=CompletionConfig)
    external_asset_manifest: Path | None = None


def _resolve_environment(value: Any) -> Any:
    if isinstance(value, str):
        match = _ENV_PATTERN.match(value)
        if match:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"required environment variable {name!r} is not set")
            # Empty optional entries are common in a sourced .env file. Treat
            # them as unset values rather than Path("."), which would make a
            # credential-only agent run look as if it had external assets.
            return os.environ[name] or None
        return value
    if isinstance(value, list):
        return [_resolve_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_environment(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> ManuscriptConfig:
    """Load YAML, resolve whole-value ``${VAR}`` references, and reject unknown keys."""

    source = Path(path)
    data = yaml.safe_load(source.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"configuration {source} must contain an object")
    return ManuscriptConfig.model_validate(_resolve_environment(data))


def apply_runtime_overrides(
    config: ManuscriptConfig,
    *,
    model: str | None = None,
    strategy_count: int | None = None,
    route_step_limit: int | None = None,
    edit_iteration_limit: int | None = None,
) -> ManuscriptConfig:
    """Return a validated copy with explicit command-line overrides applied."""

    data = config.model_dump(mode="python")
    if model is not None:
        data["llm"]["model"] = model
    if strategy_count is not None:
        data["agents"]["strategy_count"] = strategy_count
    if route_step_limit is not None:
        data["agents"]["route_step_limit"] = route_step_limit
    if edit_iteration_limit is not None:
        data["agents"]["edit_iteration_limit"] = edit_iteration_limit
    return ManuscriptConfig.model_validate(data)


def validate_external_assets(config: ManuscriptConfig) -> list[Path]:
    """Validate configured paths before an LLM call or search initializes."""

    paths: list[Path] = []
    if config.completion.enabled:
        path = config.completion.aizynthfinder_config
        if path is None:
            raise ValueError("completion is enabled but aizynthfinder_config is not set")
        paths.append(path)
    if config.external_asset_manifest is not None:
        paths.append(config.external_asset_manifest)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ValueError("missing external assets: " + ", ".join(str(path) for path in missing))
    if config.external_asset_manifest is not None:
        manifest_path = config.external_asset_manifest
        manifest = ExternalAssetManifest.model_validate_json(manifest_path.read_text())
        for asset in manifest.assets:
            asset_path = asset.path
            if not asset_path.is_absolute():
                asset_path = manifest_path.parent / asset_path
            if not asset_path.is_file():
                raise ValueError(f"missing external asset {asset.identifier!r}: {asset_path}")
            digest = hashlib.sha256()
            with asset_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != asset.sha256:
                raise ValueError(f"external asset digest mismatch: {asset.identifier}")
            paths.append(asset_path)
    return paths


__all__ = [
    "AgentConfig",
    "AssetEntry",
    "CompletionConfig",
    "LLMConfig",
    "ManuscriptConfig",
    "ExternalAssetManifest",
    "apply_runtime_overrides",
    "load_config",
    "validate_external_assets",
]
