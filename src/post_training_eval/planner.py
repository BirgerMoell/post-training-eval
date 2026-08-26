from __future__ import annotations

import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checkpoints import CheckpointRef
from .registry import load_profile, repository_root


def _command_for(step: dict[str, Any], ref: CheckpointRef, limit_override: int | None, venv_path: str | None) -> list[str] | None:
    driver = step["driver"]
    model = ref.runner_model
    limit = limit_override if limit_override is not None else step.get("limit")
    if driver == "inspect":
        return ["pteval", "inspect", "--model", model]
    if driver == "oellm-eval":
        command = ["oellm-eval", "schedule", "--models", model, "--task_groups", ",".join(step["task_groups"])]
        if limit:
            command += ["--limit", str(limit)]
        if venv_path:
            command += ["--venv_path", venv_path]
        return command
    if driver == "lm-eval":
        model_args = f"pretrained={model},dtype=bfloat16,trust_remote_code=True"
        if ref.format == "hf_hub" and ref.revision:
            model_args += f",revision={ref.revision}"
        command = ["python3", "-m", "lm_eval", "--model", "hf", "--model_args", model_args, "--tasks", ",".join(step["tasks"]), "--batch_size", str(step.get("batch_size", 1)), "--output_path", "runs/raw", "--log_samples"]
        if step.get("apply_chat_template"):
            command += ["--apply_chat_template"]
        if limit:
            command += ["--limit", str(limit)]
        return command
    if driver == "builtin-niah":
        return ["python3", "-m", "post_training_eval.niah", "--model", model, "--lengths", ",".join(str(v) for v in step["lengths"]), "--depths", ",".join(str(v) for v in step["depths"]), "--out", "runs/raw/niah.json"]
    if driver == "openai-endpoint" and ref.format == "openai_endpoint":
        endpoint, _, endpoint_model = ref.location.partition("#")
        return ["pteval", "endpoint-run", "--base-url", endpoint, "--model", endpoint_model, "--suite", step["suite"], "--out", "runs/raw/endpoint-holdouts.json"]
    return None


def build_plan(ref: CheckpointRef, profile_name: str, limit: int | None = None, venv_path: str | None = None) -> dict[str, Any]:
    profile = load_profile(profile_name)
    steps = []
    for step in profile["steps"]:
        item = dict(step)
        command = _command_for(item, ref, limit, venv_path)
        item["command"] = command
        item["command_display"] = shlex.join(command) if command else None
        item["runnable"] = command is not None and not (ref.format == "megatron" and item["driver"] not in {"inspect"})
        if ref.format == "megatron" and item["driver"] != "inspect":
            item["blocked_by"] = "Prepare the Megatron checkpoint as HF, or serve it and use openai://BASE_URL#MODEL."
        elif item["driver"] == "openai-endpoint" and ref.format != "openai_endpoint":
            item["blocked_by"] = "Serve this checkpoint behind an OpenAI-compatible endpoint, then plan with openai://BASE_URL#MODEL."
        elif item["driver"] == "oellm-eval" and ref.format == "hf_hub" and ref.revision:
            item["blocked_by"] = "oellm-eval does not currently pass Hub revisions; download the immutable snapshot and plan against its local path."
            item["runnable"] = False
        steps.append(item)
    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {"format": ref.format, "location": ref.location, "revision": ref.revision},
        "profile": {"id": profile["id"], "description": profile["description"]},
        "steps": steps,
    }


def write_plan(plan: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
