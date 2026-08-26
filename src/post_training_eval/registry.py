from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RegistryError(ValueError):
    pass


def repository_root() -> Path:
    override = Path(__import__("os").environ["PTEVAL_ROOT"]) if "PTEVAL_ROOT" in __import__("os").environ else None
    return override.resolve() if override else Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"Could not load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RegistryError(f"Expected JSON object in {path}")
    return value


def capability_manifests(root: Path | None = None) -> list[dict[str, Any]]:
    base = (root or repository_root()) / "capabilities"
    return [load_json(path) for path in sorted(base.glob("*/suite.json"))]


def load_profile(profile: str, root: Path | None = None) -> dict[str, Any]:
    path = (root or repository_root()) / "profiles" / f"{profile}.json"
    value = load_json(path)
    if value.get("id") != profile:
        raise RegistryError(f"Profile id in {path} does not match filename")
    return value


def benchmark_index(root: Path | None = None) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for capability in capability_manifests(root):
        capability_id = capability["id"]
        for benchmark in capability.get("benchmarks", []):
            item = dict(benchmark)
            item["capability"] = capability_id
            if item["id"] in index:
                raise RegistryError(f"Duplicate benchmark id: {item['id']}")
            index[item["id"]] = item
    return index


def validate_registry(root: Path | None = None) -> list[str]:
    base = root or repository_root()
    errors: list[str] = []
    capabilities = capability_manifests(base)
    ids = [item.get("id") for item in capabilities]
    if len(ids) != len(set(ids)):
        errors.append("Capability ids are not unique")
    required = {"safety", "multilingual", "instruction-chat", "reasoning-knowledge", "grounding-rag", "tools-agents", "long-context", "coding", "efficiency-release"}
    missing = sorted(required - set(ids))
    if missing:
        errors.append(f"Missing capabilities: {', '.join(missing)}")
    try:
        benchmark_index(base)
    except RegistryError as exc:
        errors.append(str(exc))
    for profile_path in sorted((base / "profiles").glob("*.json")):
        profile = load_json(profile_path)
        for step in profile.get("steps", []):
            if step.get("capability") not in ids:
                errors.append(f"{profile_path.name}: unknown capability {step.get('capability')}")
            if step.get("driver") not in {"inspect", "oellm-eval", "lm-eval", "builtin-niah", "external", "openai-endpoint"}:
                errors.append(f"{profile_path.name}: unknown driver {step.get('driver')}")
    return errors

