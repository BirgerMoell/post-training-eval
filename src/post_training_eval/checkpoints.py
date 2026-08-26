from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CheckpointError(ValueError):
    pass


@dataclass(frozen=True)
class CheckpointRef:
    format: str
    location: str
    revision: str | None = None

    @property
    def runner_model(self) -> str:
        if self.format == "hf_hub":
            return self.location
        return self.location


def parse_checkpoint(value: str) -> CheckpointRef:
    if value.startswith("hf://"):
        raw = value[5:]
        model_id, sep, revision = raw.partition("@")
        if model_id.count("/") != 1:
            raise CheckpointError("HF references must be hf://owner/model[@revision]")
        return CheckpointRef("hf_hub", model_id, revision if sep else None)
    if value.startswith("megatron://"):
        return CheckpointRef("megatron", value[len("megatron://") :])
    if value.startswith("openai://"):
        return CheckpointRef("openai_endpoint", value[len("openai://") :])
    candidate = Path(value).expanduser()
    if candidate.exists():
        return CheckpointRef("megatron" if _looks_like_megatron(candidate) else "hf_local", str(candidate.resolve()))
    if value.count("/") == 1:
        return CheckpointRef("hf_hub", value)
    raise CheckpointError(f"Cannot infer checkpoint format from {value!r}")


def _looks_like_megatron(path: Path) -> bool:
    if (path / "latest_checkpointed_iteration.txt").exists():
        return True
    return any(path.glob("iter_*/mp_rank_*")) or any(path.glob("mp_rank_*"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"Could not read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"Expected an object in {path}")
    return value


def _fetch_json(url: str, timeout: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "openeurollm-post-training-eval/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except Exception as exc:
        raise CheckpointError(f"Could not fetch {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise CheckpointError(f"Expected an object from {url}")
    return value


def _architecture(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures") or [],
        "max_position_embeddings": config.get("max_position_embeddings"),
        "rope_theta": config.get("rope_theta") or (config.get("rope_parameters") or {}).get("rope_theta"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "vocab_size": config.get("vocab_size"),
        "torch_dtype": config.get("torch_dtype") or config.get("dtype"),
    }


def inspect_checkpoint(ref: CheckpointRef) -> dict[str, Any]:
    if ref.format == "hf_hub":
        revision = ref.revision or "main"
        quoted = urllib.parse.quote(ref.location, safe="/")
        api_url = f"https://huggingface.co/api/models/{quoted}/revision/{urllib.parse.quote(revision)}"
        raw_url = f"https://huggingface.co/{quoted}/resolve/{urllib.parse.quote(revision)}/config.json"
        metadata = _fetch_json(api_url)
        config = _fetch_json(raw_url)
        siblings = [item.get("rfilename", "") for item in metadata.get("siblings", [])]
        shards = sorted(name for name in siblings if re.search(r"\.safetensors$", name))
        safetensors = metadata.get("safetensors") or {}
        return {
            "schema_version": 1,
            "status": "compatible" if shards else "incomplete",
            "format": "hf_hub",
            "model_id": ref.location,
            "requested_revision": revision,
            "resolved_revision": metadata.get("sha"),
            "source_url": f"https://huggingface.co/{ref.location}/tree/{metadata.get('sha') or revision}",
            "private": bool(metadata.get("private")),
            "gated": metadata.get("gated", False),
            "library_name": metadata.get("library_name"),
            "pipeline_tag": metadata.get("pipeline_tag"),
            "license": (metadata.get("cardData") or {}).get("license"),
            "architecture": _architecture(config),
            "weight_files": shards,
            "parameter_count": safetensors.get("total"),
            "chat_template_present": "chat_template.jinja" in siblings or "tokenizer_config.json" in siblings,
            "checks": {
                "config_json": "config.json" in siblings,
                "tokenizer": any(name in siblings for name in ("tokenizer.json", "tokenizer.model")),
                "generation_config": "generation_config.json" in siblings,
                "safetensors_index": "model.safetensors.index.json" in siblings or len(shards) == 1,
                "weights": bool(shards),
            },
        }
    if ref.format == "hf_local":
        path = Path(ref.location)
        config = _read_json(path / "config.json")
        shards = sorted(item.name for item in path.glob("*.safetensors"))
        checks = {
            "config_json": True,
            "tokenizer": (path / "tokenizer.json").exists() or (path / "tokenizer.model").exists(),
            "generation_config": (path / "generation_config.json").exists(),
            "safetensors_index": (path / "model.safetensors.index.json").exists() or len(shards) == 1,
            "weights": bool(shards),
        }
        return {
            "schema_version": 1,
            "status": "compatible" if all((checks["config_json"], checks["tokenizer"], checks["weights"])) else "incomplete",
            "format": "hf_local",
            "model_id": path.name,
            "path": str(path),
            "architecture": _architecture(config),
            "weight_files": shards,
            "chat_template_present": (path / "chat_template.jinja").exists() or "chat_template" in _read_json(path / "tokenizer_config.json") if (path / "tokenizer_config.json").exists() else False,
            "checks": checks,
        }
    if ref.format == "megatron":
        path = Path(ref.location).expanduser().resolve()
        tracker = path / "latest_checkpointed_iteration.txt"
        iteration = tracker.read_text().strip() if tracker.exists() else None
        iteration_dir = path / f"iter_{int(iteration):07d}" if iteration and iteration.isdigit() else path
        rank_dirs = sorted(str(item.relative_to(path)) for item in iteration_dir.glob("mp_rank_*"))
        tensor_files = sorted(str(item.relative_to(path)) for item in iteration_dir.glob("mp_rank_*/*.pt"))
        return {
            "schema_version": 1,
            "status": "conversion_required" if rank_dirs and tensor_files else "incomplete",
            "format": "megatron",
            "model_id": path.name,
            "path": str(path),
            "iteration": int(iteration) if iteration and iteration.isdigit() else iteration,
            "rank_directories": rank_dirs,
            "tensor_files": tensor_files,
            "checks": {"tracker": tracker.exists(), "rank_directories": bool(rank_dirs), "tensor_files": bool(tensor_files)},
        }
    if ref.format == "openai_endpoint":
        endpoint, sep, model = ref.location.partition("#")
        if not sep or not endpoint or not model:
            raise CheckpointError("Endpoint references must be openai://BASE_URL#MODEL")
        return {"schema_version": 1, "status": "endpoint_declared", "format": ref.format, "model_id": model, "base_url": endpoint}
    raise CheckpointError(f"Unsupported checkpoint format: {ref.format}")


def prepare_megatron(config_path: Path, execute: bool) -> dict[str, Any]:
    config = _read_json(config_path)
    ref = parse_checkpoint(config.get("checkpoint", ""))
    if ref.format != "megatron":
        raise CheckpointError("Megatron preparation config must point to megatron://...")
    output = Path(config.get("output", "")).expanduser()
    command = config.get("converter", {}).get("command")
    if not isinstance(command, list) or not command or not all(isinstance(part, str) for part in command):
        raise CheckpointError("converter.command must be a non-empty JSON array; shell strings are not accepted")
    replacements = {"{source}": ref.location, "{output}": str(output), "{tokenizer}": str(config.get("tokenizer", ""))}
    resolved = [replacements.get(part, part) for part in command]
    report: dict[str, Any] = {"source": ref.location, "output": str(output), "command": resolved, "executed": execute}
    if execute:
        output.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(resolved, check=False)
        report["returncode"] = completed.returncode
        if completed.returncode:
            raise CheckpointError(f"Converter exited with {completed.returncode}")
        report["prepared_checkpoint"] = inspect_checkpoint(CheckpointRef("hf_local", str(output)))
    return report
