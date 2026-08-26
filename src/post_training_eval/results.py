from __future__ import annotations

import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .registry import benchmark_index, capability_manifests, load_json, repository_root


class ResultError(ValueError):
    pass


PREFERRED_METRICS = (
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "prompt_level_strict_acc,none",
    "inst_level_strict_acc,none",
    "acc_norm,none",
    "acc,none",
    "exact_match,none",
    "pass@1,none",
)


def validate_result(run: dict[str, Any]) -> list[str]:
    errors = []
    for key in ("schema_version", "run_id", "status", "model", "provenance", "metrics"):
        if key not in run:
            errors.append(f"missing {key}")
    if not isinstance(run.get("metrics"), list):
        errors.append("metrics must be a list")
        return errors
    for index, metric in enumerate(run["metrics"]):
        for key in ("capability", "benchmark", "metric", "value", "scale"):
            if key not in metric:
                errors.append(f"metrics[{index}] missing {key}")
        if metric.get("scale") == "percentage" and isinstance(metric.get("value"), (int, float)) and not 0 <= metric["value"] <= 100:
            errors.append(f"metrics[{index}] percentage outside 0..100")
    return errors


def _task_to_benchmark(task: str) -> str:
    if task.startswith("mgsm_native_cot_"):
        return "mgsm"
    if task.startswith("global_mmlu"):
        return "global-mmlu"
    if task.startswith("mmlu_prox"):
        return "mmlu-prox"
    if task.startswith("polymath"):
        return "polymath"
    if task.startswith("sib200"):
        return "sib-200"
    if task.startswith("belebele"):
        return "belebele"
    mapping = {"ifeval": "ifeval", "gsm8k": "gsm8k", "arc_challenge": "arc-challenge", "mbpp": "mbpp", "humaneval": "humaneval", "HumanEval": "humaneval", "LiveCodeBench": "livecodebench"}
    return mapping.get(task, task.replace("_", "-"))


def ingest_lm_eval(input_path: Path, model_id: str, run_id: str, model_revision: str | None, source_command: str | None) -> dict[str, Any]:
    raw = load_json(input_path)
    registry = benchmark_index()
    metrics: list[dict[str, Any]] = []
    for task, values in (raw.get("results") or {}).items():
        if not isinstance(values, dict):
            continue
        benchmark_id = _task_to_benchmark(task)
        benchmark = registry.get(benchmark_id, {})
        for metric_name in PREFERRED_METRICS:
            value = values.get(metric_name)
            if isinstance(value, (int, float)):
                language = task.rsplit("_", 1)[-1] if benchmark_id in {"mgsm", "global-mmlu", "mmlu-prox", "sib-200", "belebele", "polymath"} else None
                metrics.append({
                    "capability": benchmark.get("capability", "unmapped"),
                    "benchmark": benchmark_id,
                    "task": task,
                    "metric": metric_name,
                    "value": round(float(value) * 100, 6),
                    "scale": "percentage",
                    "direction": benchmark.get("direction", "higher"),
                    "language": language,
                    "n": (raw.get("n-samples") or {}).get(task, {}).get("effective"),
                })
                break
    now = datetime.now(timezone.utc).isoformat()
    run = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed" if metrics else "failed",
        "started_at": raw.get("date") or now,
        "finished_at": now,
        "model": {"id": model_id, "revision": model_revision, "format": "hf"},
        "provenance": {"kind": "fresh-reproduced", "source": str(input_path), "command": source_command, "harness": "lm-evaluation-harness", "harness_version": (raw.get("versions") or {}).get("lm_eval")},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "metrics": metrics,
    }
    errors = validate_result(run)
    if errors:
        raise ResultError("; ".join(errors))
    return run


def publish_run(source: Path, root: Path | None = None) -> Path:
    base = root or repository_root()
    run = load_json(source)
    errors = validate_result(run)
    if errors:
        raise ResultError("; ".join(errors))
    destination = base / "results" / "runs" / f"{run['run_id']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    return destination


def build_site_data(root: Path | None = None) -> dict[str, Any]:
    base = root or repository_root()
    capabilities = capability_manifests(base)
    runs = []
    for path in sorted((base / "results" / "runs").glob("*.json")):
        run = load_json(path)
        errors = validate_result(run)
        if errors:
            raise ResultError(f"{path}: {'; '.join(errors)}")
        runs.append(run)
    gates = [load_json(path) for path in sorted((base / "results" / "gates").glob("*.json"))]
    models: dict[str, dict[str, Any]] = {}
    for run in runs:
        model = run["model"]
        entry = models.setdefault(model["id"], {"id": model["id"], "revisions": set(), "run_ids": [], "capabilities": set(), "metric_count": 0})
        if model.get("revision"):
            entry["revisions"].add(model["revision"])
        entry["run_ids"].append(run["run_id"])
        entry["metric_count"] += len(run["metrics"])
        entry["capabilities"].update(metric["capability"] for metric in run["metrics"])
    serial_models = []
    for model in models.values():
        serial_models.append({**model, "revisions": sorted(model["revisions"]), "capabilities": sorted(model["capabilities"])})
    generated_at = max((run.get("finished_at") or run.get("started_at") or "1970-01-01T00:00:00Z" for run in runs), default="1970-01-01T00:00:00Z")
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "capabilities": capabilities,
        "models": sorted(serial_models, key=lambda item: item["id"]),
        "runs": runs,
        "gates": gates,
    }


def write_site_data(root: Path | None = None) -> Path:
    base = root or repository_root()
    destination = base / "docs" / "data" / "index.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(build_site_data(base), indent=2, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    return destination


def git_publish(root: Path, run_id: str) -> None:
    subprocess.run(["git", "add", "results/runs", "docs/data/index.json"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", f"results: publish {run_id}"], cwd=root, check=True)
    subprocess.run(["git", "push"], cwd=root, check=True)
