from __future__ import annotations

import hashlib
import subprocess
import sys

import pytest
from pydantic import ValidationError

from synthex.config import (
    ExternalAssetManifest,
    ManuscriptConfig,
    apply_runtime_overrides,
    load_config,
    validate_external_assets,
)


def test_defaults_match_manuscript_method():
    config = ManuscriptConfig(completion={"enabled": False})
    assert config.llm.model == "gemini-3.1-pro-preview"
    assert not config.llm.use_vertex_ai
    assert config.llm.timeout_seconds == 600
    assert config.llm.max_attempts == 3
    assert config.agents.route_step_limit == 25
    assert config.agents.edit_iteration_limit == 6
    assert (
        config.agents.strategy_temperature,
        config.agents.route_temperature,
        config.agents.critic_temperature,
        config.agents.editor_temperature,
        config.agents.analyst_temperature,
    ) == (0.1, 0.3, 0.1, 0.3, 0.3)
    assert config.agents.route_thinking_level == "medium"
    assert config.agents.critic_thinking_level == "medium"
    assert config.agents.editor_thinking_level == "medium"
    assert config.completion.budget.model_dump() == {
        "max_transforms": 6,
        "iteration_limit": 500,
        "time_limit": 1200,
        "return_first": True,
    }


def test_unknown_config_keys_are_rejected():
    with pytest.raises(ValidationError):
        ManuscriptConfig.model_validate({"completion": {"enabled": False}, "unused": 1})


def test_runtime_overrides_are_validated_and_do_not_mutate_source():
    source = ManuscriptConfig(completion={"enabled": False})
    overridden = apply_runtime_overrides(
        source,
        model="gemini-3.7-flash",
        strategy_count=1,
        route_step_limit=8,
        edit_iteration_limit=0,
    )
    assert source.llm.model == "gemini-3.1-pro-preview"
    assert source.agents.edit_iteration_limit == 6
    assert overridden.llm.model == "gemini-3.7-flash"
    assert overridden.agents.strategy_count == 1
    assert overridden.agents.route_step_limit == 8
    assert overridden.agents.edit_iteration_limit == 0

    with pytest.raises(ValidationError):
        apply_runtime_overrides(source, edit_iteration_limit=-1)


def test_empty_optional_environment_path_resolves_to_none(tmp_path, monkeypatch):
    monkeypatch.setenv("EMPTY_SYNTHEX_TEST_PATH", "")
    path = tmp_path / "config.yml"
    path.write_text(
        "format_version: synthex-config-1.0\n"
        "completion:\n"
        "  enabled: false\n"
        "  aizynthfinder_config: ${EMPTY_SYNTHEX_TEST_PATH}\n"
    )
    assert load_config(path).completion.aizynthfinder_config is None


def test_external_asset_identifiers_are_unique():
    with pytest.raises(ValidationError, match="identifiers must be unique"):
        ExternalAssetManifest.model_validate(
            {
                "assets": [
                    {"identifier": "stock", "path": "a", "sha256": "0" * 64},
                    {"identifier": "stock", "path": "b", "sha256": "1" * 64},
                ]
            }
        )


def test_missing_environment_variable_is_clear(tmp_path, monkeypatch):
    monkeypatch.delenv("MISSING_SYNTHEX_TEST_PATH", raising=False)
    path = tmp_path / "config.yml"
    path.write_text(
        "format_version: synthex-config-1.0\n"
        "completion:\n"
        "  enabled: true\n"
        "  aizynthfinder_config: ${MISSING_SYNTHEX_TEST_PATH}\n"
    )
    with pytest.raises(ValueError, match="MISSING_SYNTHEX_TEST_PATH"):
        load_config(path)


def test_parsing_config_does_not_import_planner_or_gemini_sdk():
    code = """
import sys
import synthex.config
assert 'aizynthfinder.aizynthfinder' not in sys.modules
assert 'google.genai' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_external_asset_manifest_verifies_digests(tmp_path):
    asset = tmp_path / "stock.bin"
    asset.write_bytes(b"fixture stock")
    aiz_config = tmp_path / "aiz.yml"
    aiz_config.write_text("search: {}\n")
    manifest = tmp_path / "assets.json"
    manifest.write_text(
        "{\n"
        '  "format_version": "external-assets-1.0",\n'
        '  "assets": [{"identifier": "stock", "path": "stock.bin", '
        f'"sha256": "{hashlib.sha256(asset.read_bytes()).hexdigest()}"}}]\n'
        "}\n"
    )
    config = ManuscriptConfig(
        completion={"aizynthfinder_config": aiz_config},
        external_asset_manifest=manifest,
    )
    assert validate_external_assets(config) == [aiz_config, manifest, asset]
