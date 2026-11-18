"""Versioned run and external-benchmark artifact contracts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def digest_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def now_utc() -> str:
    return datetime.now(UTC).isoformat()


def redact_sensitive_text(value: str) -> str:
    """Remove credential-shaped values before writing errors or CLI output."""

    redacted = value
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        secret = os.environ.get(name)
        if secret and len(secret) >= 8:
            redacted = redacted.replace(secret, "[REDACTED]")
    redacted = re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED]", redacted)
    return re.sub(
        r"(?i)\b(api[_-]?key|key|access[_-]?token)=([^\s&]+)",
        r"\1=[REDACTED]",
        redacted,
    )


def code_revision() -> str:
    """Return the Git revision with an explicit dirty suffix when discoverable."""

    source_root = Path(__file__).resolve().parents[2]
    try:
        package_fallback = f"package:{version('synthex')}"
    except PackageNotFoundError:
        package_fallback = "unavailable"
    if not (source_root / ".git").exists():
        return package_fallback
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=source_root,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=source_root,
        ).stdout
        return revision + ("+dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return package_fallback


def dependency_versions() -> dict[str, str]:
    output: dict[str, str] = {}
    for package in ("synthex", "aizynthfinder", "google-genai", "pydantic", "rdkit"):
        try:
            output[package] = version(package)
        except PackageNotFoundError:
            output[package] = "not-installed"
    return output


class ArtifactModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ArtifactReference(ArtifactModel):
    path: str
    sha256: str
    parent_sha256: str | None = None


class Failure(ArtifactModel):
    stage: str
    category: str
    message: str


class RunManifest(ArtifactModel):
    format_version: Literal["run-manifest-1.0"] = "run-manifest-1.0"
    run_id: str
    target: str
    target_id: str | None = None
    constraint: str | None = None
    input_sha256: str
    config: dict[str, Any]
    config_sha256: str
    code_revision: str
    dependency_versions: dict[str, str]
    dependency_lock_sha256: str | None = None
    provider: str
    model: str
    prompt_versions: dict[str, str]
    prompt_sha256: dict[str, str]
    external_assets: dict[str, str] = Field(default_factory=dict)
    # The top-level config is the resolved configuration at run creation.
    # Staged runs may intentionally change settings for unfinished stages;
    # every checkpoint therefore records its own exact resolved configuration.
    stage_configs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    stage_config_sha256: dict[str, str] = Field(default_factory=dict)
    llm_usage: dict[str, int] = Field(default_factory=dict)
    stage_durations_seconds: dict[str, float] = Field(default_factory=dict)
    status: Literal["running", "partial", "complete", "failed"] = "running"
    current_stage: str = "created"
    completed_stages: list[str] = Field(default_factory=list)
    artifacts: dict[str, ArtifactReference] = Field(default_factory=dict)
    failure: Failure | None = None
    created_at: str
    updated_at: str


class BenchmarkTarget(ArtifactModel):
    format_version: Literal["benchmark-target-1.0"] = "benchmark-target-1.0"
    target_id: str
    smiles: str
    constraint: str | None = None


class BenchmarkResult(ArtifactModel):
    format_version: Literal["benchmark-result-1.0"] = "benchmark-result-1.0"
    target_id: str
    status: Literal["complete", "failed"]
    run_manifest: str
    strategic_routes: str | None = None
    stitched_routes: str | None = None
    failure_category: str | None = None


class ArtifactStore:
    """Immutable checkpoint writer plus mutable, validated run manifest."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def load_manifest(self) -> RunManifest:
        return RunManifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))

    def save_manifest(self, manifest: RunManifest) -> None:
        manifest.updated_at = now_utc()
        temporary = self.root / ".manifest.json.tmp"
        # Pin encoding and newlines: digest-linked artifacts must be byte-stable
        # across platforms (Windows otherwise translates LF to CRLF).
        temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(self.manifest_path)

    def write_checkpoint(
        self, name: str, payload: Any, parent_sha256: str | None = None
    ) -> ArtifactReference:
        filename = f"{name}.json"
        path = self.root / filename
        text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
        # Compute the digest from the exact bytes written. Using write_text with
        # platform defaults would translate LF to CRLF on Windows and encode
        # non-ASCII text with the locale codec, so digest_file() would never
        # match the reference recorded in the manifest.
        data = text.encode("utf-8")
        digest = hashlib.sha256(data).hexdigest()
        if path.exists():
            existing_digest = digest_file(path)
            if existing_digest != digest:
                raise ValueError(
                    f"immutable checkpoint {filename} already exists with different content"
                )
        else:
            path.write_bytes(data)
        return ArtifactReference(path=filename, sha256=digest, parent_sha256=parent_sha256)

    def read_checkpoint(self, reference: ArtifactReference) -> Any:
        path = self.root / reference.path
        if digest_file(path) != reference.sha256:
            raise ValueError(f"checkpoint digest mismatch: {reference.path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def validate(self) -> RunManifest:
        manifest = self.load_manifest()
        if digest_json(manifest.config) != manifest.config_sha256:
            raise ValueError("manifest configuration digest mismatch")
        if len(manifest.completed_stages) != len(set(manifest.completed_stages)):
            raise ValueError("manifest completed_stages contains duplicates")
        if set(manifest.artifacts) != set(manifest.completed_stages):
            raise ValueError("manifest artifacts and completed_stages do not match")
        if set(manifest.stage_configs) != set(manifest.completed_stages):
            raise ValueError("manifest stage_configs and completed_stages do not match")
        if set(manifest.stage_config_sha256) != set(manifest.completed_stages):
            raise ValueError("manifest stage_config_sha256 and completed_stages do not match")
        parent_sha256: str | None = None
        for stage in manifest.completed_stages:
            if digest_json(manifest.stage_configs[stage]) != manifest.stage_config_sha256[stage]:
                raise ValueError(f"stage configuration digest mismatch: {stage}")
            reference = manifest.artifacts[stage]
            if reference.parent_sha256 != parent_sha256:
                raise ValueError(f"checkpoint parent digest mismatch: {stage}")
            self.read_checkpoint(reference)
            parent_sha256 = reference.sha256
        return manifest


def run_manifest_schema() -> dict[str, Any]:
    return RunManifest.model_json_schema()


def benchmark_interface_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SynthEx benchmark interface 1.0",
        "$defs": {
            "BenchmarkTarget": BenchmarkTarget.model_json_schema(),
            "BenchmarkResult": BenchmarkResult.model_json_schema(),
        },
    }


__all__ = [
    "ArtifactReference",
    "ArtifactStore",
    "BenchmarkResult",
    "BenchmarkTarget",
    "Failure",
    "RunManifest",
    "benchmark_interface_schema",
    "canonical_json",
    "code_revision",
    "dependency_versions",
    "digest_file",
    "digest_json",
    "now_utc",
    "redact_sensitive_text",
    "run_manifest_schema",
]
