from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


class LocalQuickError(ValueError):
    pass


@dataclass(frozen=True)
class TaskSpec:
    task: str
    n_shot: int
    suite: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chunks(values: Sequence[TaskSpec], size: int) -> list[list[TaskSpec]]:
    if size < 1:
        raise LocalQuickError("max_tasks_per_invocation must be at least 1")
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def batch_tasks(tasks: Iterable[TaskSpec], max_tasks_per_invocation: int = 32) -> list[list[TaskSpec]]:
    """Group only tasks with the same harness and few-shot protocol."""
    groups: dict[tuple[str, int], list[TaskSpec]] = {}
    for task in tasks:
        groups.setdefault((task.suite, task.n_shot), []).append(task)
    batches: list[list[TaskSpec]] = []
    suite_order = {"lm-eval-harness": 0, "lighteval": 1, "evalchemy": 2}
    for key in sorted(groups, key=lambda item: (suite_order.get(item[0], 99), item[1])):
        batches.extend(_chunks(sorted(groups[key], key=lambda item: item.task), max_tasks_per_invocation))
    return batches


def load_oellm_tasks() -> list[TaskSpec]:
    try:
        from oellm.task_groups import _expand_task_groups
    except ImportError as exc:
        raise LocalQuickError(
            "OpenEuroLLM/oellm-eval is required. Install it or add its checkout to PYTHONPATH."
        ) from exc
    return [TaskSpec(row.task, int(row.n_shot), row.suite) for row in _expand_task_groups(["all"])]


def build_lm_eval_command(
    python: str,
    model: str,
    tasks: Sequence[TaskSpec],
    output: Path,
    include_path: Path,
    limit: int,
) -> list[str]:
    if not tasks or len({task.n_shot for task in tasks}) != 1:
        raise LocalQuickError("lm-eval batches must be non-empty and use one n-shot value")
    return [
        python,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model},trust_remote_code=True,dtype=bfloat16",
        "--tasks",
        ",".join(task.task for task in tasks),
        "--num_fewshot",
        str(tasks[0].n_shot),
        "--output_path",
        str(output),
        "--include_path",
        str(include_path),
        "--batch_size",
        "1",
        "--limit",
        str(limit),
        "--apply_chat_template",
        "--log_samples",
    ]


def build_lighteval_command(
    executable: str,
    model: str,
    tasks: Sequence[TaskSpec],
    output: Path,
    limit: int,
) -> list[str]:
    if not tasks:
        raise LocalQuickError("lighteval batches must be non-empty")
    task_argument = ",".join(f"{task.task}|{task.n_shot}" for task in tasks)
    return [
        executable,
        "accelerate",
        f"model_name={model},trust_remote_code=True,batch_size=1",
        task_argument,
        "--load-tasks-multilingual",
        "--output-dir",
        str(output),
        "--max-samples",
        str(limit),
    ]


def build_evalchemy_command(
    python: str,
    model: str,
    tasks: Sequence[TaskSpec],
    output: Path,
    limit: int,
    tokenizer: str | None = None,
) -> list[str]:
    if not tasks:
        raise LocalQuickError("Evalchemy batches must be non-empty")
    model_args = f"trust_remote_code=True,pretrained={model},dtype=bfloat16"
    if tokenizer:
        model_args += f",tokenizer={tokenizer}"
    return [
        str(Path(python).with_name("accelerate")),
        "launch",
        "--num-processes",
        "1",
        "--num-machines",
        "1",
        "-m",
        "eval.eval",
        "--model",
        "hf",
        "--tasks",
        ",".join(task.task for task in tasks),
        "--model_args",
        model_args,
        "--batch_size",
        "1",
        "--output_path",
        str(output),
        "--limit",
        str(limit),
        "--apply_chat_template",
        "--log_samples",
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def run_local_quick(
    *,
    model: str,
    output_dir: Path,
    lm_python: str,
    include_path: Path,
    lighteval_bin: str | None = None,
    evalchemy_python: str | None = None,
    evalchemy_dir: Path | None = None,
    evalchemy_tokenizer: str | None = None,
    suites: Sequence[str] = ("lm-eval-harness", "lighteval", "evalchemy"),
    limit: int = 8,
    gpu: str = "0",
    max_tasks_per_invocation: int = 32,
    min_free_gb: float = 20,
    hf_home: str | None = None,
) -> Path:
    if limit < 1:
        raise LocalQuickError("limit must be at least 1")
    requested = set(suites)
    known = {"lm-eval-harness", "lighteval", "evalchemy"}
    if requested - known:
        raise LocalQuickError(f"Unknown suites: {', '.join(sorted(requested - known))}")
    if "lighteval" in requested and not lighteval_bin:
        raise LocalQuickError("--lighteval-bin is required for the lighteval suite")
    if "evalchemy" in requested and (not evalchemy_python or not evalchemy_dir):
        raise LocalQuickError("--evalchemy-python and --evalchemy-dir are required for Evalchemy")

    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [task for task in load_oellm_tasks() if task.suite in requested]
    batches = batch_tasks(tasks, max_tasks_per_invocation)
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, Any]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("model") != model or manifest.get("limit") != limit:
            raise LocalQuickError("Existing manifest belongs to a different model or limit")
        existing_suites = manifest.get("suites") or sorted({task["suite"] for task in manifest.get("tasks", [])})
        if sorted(existing_suites) != sorted(requested):
            raise LocalQuickError("Existing manifest was created for a different harness selection")
    else:
        manifest = {
            "schema_version": 1,
            "profile": "quick",
            "diagnostic": True,
            "model": model,
            "limit": limit,
            "gpu": gpu,
            "suites": sorted(requested),
            "started_at": _now(),
            "tasks": [asdict(task) for task in tasks],
            "batches": {},
        }

    with (output_dir / "jobs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("model_path", "task_path", "n_shot", "eval_suite"))
        writer.writeheader()
        for task in tasks:
            writer.writerow({"model_path": model, "task_path": task.task, "n_shot": task.n_shot, "eval_suite": task.suite})

    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": gpu,
        "TOKENIZERS_PARALLELISM": "false",
        "HF_ALLOW_CODE_EVAL": "1",
    })
    if hf_home:
        environment["HF_HOME"] = hf_home

    for index, batch in enumerate(batches):
        suite = batch[0].suite
        batch_id = f"{suite}-shot{batch[0].n_shot}-{index:03d}"
        existing = manifest["batches"].get(batch_id, {})
        if existing.get("status") == "completed":
            continue
        batch_dir = output_dir / "raw" / batch_id
        batch_dir.mkdir(parents=True, exist_ok=True)
        if _free_gb(output_dir) < min_free_gb:
            raise LocalQuickError(f"Free disk fell below {min_free_gb:g} GiB before {batch_id}")
        if suite == "lm-eval-harness":
            command = build_lm_eval_command(lm_python, model, batch, batch_dir, include_path, limit)
            cwd = None
        elif suite == "lighteval":
            command = build_lighteval_command(str(lighteval_bin), model, batch, batch_dir, limit)
            cwd = None
        else:
            command = build_evalchemy_command(str(evalchemy_python), model, batch, batch_dir, limit, evalchemy_tokenizer)
            cwd = evalchemy_dir
        manifest["batches"][batch_id] = {
            "suite": suite,
            "n_shot": batch[0].n_shot,
            "tasks": [task.task for task in batch],
            "status": "running",
            "started_at": _now(),
            "command": command,
            "log": str(batch_dir / "runner.log"),
        }
        _write_json(manifest_path, manifest)
        with (batch_dir / "runner.log").open("a") as log:
            completed = subprocess.run(command, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
        manifest["batches"][batch_id].update({
            "status": "completed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "finished_at": _now(),
        })
        _write_json(manifest_path, manifest)
        if completed.returncode:
            raise LocalQuickError(f"{batch_id} failed; see {batch_dir / 'runner.log'}")

    manifest["status"] = "completed"
    manifest["finished_at"] = _now()
    _write_json(manifest_path, manifest)
    return manifest_path
