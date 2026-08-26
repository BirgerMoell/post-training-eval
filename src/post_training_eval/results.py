from __future__ import annotations

import csv
import json
import platform
import re
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
    "f1,none",
    "bleu,none",
    "chrf++,none",
)


def validate_result(run: dict[str, Any]) -> list[str]:
    errors = []
    for key in ("schema_version", "run_id", "status", "model", "provenance", "metrics"):
        if key not in run:
            errors.append(f"missing {key}")
    if not isinstance(run.get("metrics"), list):
        errors.append("metrics must be a list")
        return errors
    registry = benchmark_index()
    for index, metric in enumerate(run["metrics"]):
        for key in ("capability", "benchmark", "metric", "value", "scale"):
            if key not in metric:
                errors.append(f"metrics[{index}] missing {key}")
        if metric.get("scale") == "percentage" and isinstance(metric.get("value"), (int, float)) and not 0 <= metric["value"] <= 100:
            errors.append(f"metrics[{index}] percentage outside 0..100")
        benchmark = registry.get(metric.get("benchmark"))
        if benchmark is None:
            errors.append(f"metrics[{index}] unknown benchmark {metric.get('benchmark')}")
        elif benchmark.get("capability") != metric.get("capability"):
            errors.append(f"metrics[{index}] benchmark {metric.get('benchmark')} belongs to {benchmark.get('capability')}, not {metric.get('capability')}")
    return errors


def _task_to_benchmark(task: str) -> str:
    if task.startswith(("mgsm_native_cot_", "global_mgsm_")):
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
    if task.startswith("flores200:"):
        return "flores-200"
    if task.startswith("include_base"):
        return "include"
    if task.startswith("xcsqa_"):
        return "xcsqa"
    if task.startswith("pawsx_"):
        return "paws-x"
    if task.startswith("xnli_"):
        return "xnli"
    if task.startswith("opensubtitles_multi40_"):
        return "opensubtitles-multi40"
    if task.startswith("arc_challenge_mt_"):
        return "arc-challenge-mt"
    if task.startswith(("global_piqa_completions_", "global_piqa_prompted_")):
        return "global-piqa"
    if task in {"copa", "social_iqa", "openbookqa", "lambada_openai", "winogrande", "mmlu", "hellaswag", "arc_easy", "commonsense_qa", "piqa", "boolq"}:
        return "open-sci"
    if task in {"agieval_lsat_ar", "wsc273", "bigbench_language_identification_multiple_choice", "bigbench_qa_wikidata_generate_until", "bigbench_dyck_languages_generate_until", "bigbench_operators_generate_until", "bigbench_repeat_copy_logic_generate_until", "bigbench_cs_algorithms_generate_until", "coqa", "squadv2", "jeopardy"}:
        return "dclm-core"
    mapping = {"ifeval": "ifeval", "gsm8k": "gsm8k", "arc_challenge": "arc-challenge", "mbpp": "mbpp", "humaneval": "humaneval", "HumanEval": "humaneval", "LiveCodeBench": "livecodebench", "mmlu_college_computer_science": "mmlu-college-cs", "hendrycks_math500": "math-500", "MATH500": "math-500", "GPQADiamond": "gpqa-diamond", "AIME24": "aime24", "AIME25": "aime25", "AMC23": "amc23", "xwinograd": "xwinograd", "xcopa": "xcopa", "xstorycloze": "xstorycloze"}
    return mapping.get(task, task.replace("_", "-"))


def _task_language(task: str, benchmark_id: str) -> str | None:
    prefixes = {
        "mgsm": ("mgsm_native_cot_", "global_mgsm_"),
        "global-mmlu": ("global_mmlu_full_",),
        "mmlu-prox": ("mmlu_prox_",),
        "sib-200": ("sib200_",),
        "belebele": ("belebele_",),
        "xcsqa": ("xcsqa_",),
        "paws-x": ("pawsx_",),
        "xnli": ("xnli_",),
        "arc-challenge-mt": ("arc_challenge_mt_",),
        "global-piqa": ("global_piqa_completions_", "global_piqa_prompted_"),
    }
    for prefix in prefixes.get(benchmark_id, ()):
        if task.startswith(prefix):
            return task[len(prefix) :].removesuffix("_cf")
    if benchmark_id == "polymath" and task.startswith("polymath_"):
        return task.split("_")[1]
    if benchmark_id == "flores-200":
        return task.split(":", 1)[-1]
    if benchmark_id == "include" and task.startswith("include_base_44_"):
        return task[len("include_base_44_") :]
    if benchmark_id == "opensubtitles-multi40":
        return task[len("opensubtitles_multi40_") :]
    return None


def _normal_value(value: float, metric_name: str) -> tuple[float, str]:
    metric_lower = metric_name.lower()
    if "chrf" in metric_lower or "bleu" in metric_lower:
        return value, "score"
    return (value * 100 if -1 <= value <= 1 else value), "percentage"


def _iso_timestamp(value: Any, fallback: str) -> str:
    """Normalize lm-eval's numeric or textual run date for stable site sorting."""
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc).isoformat()
    if isinstance(value, str) and value.strip():
        return value
    return fallback


def _ingest_lm_eval_raw(raw: dict[str, Any], source: str, model_id: str, run_id: str, model_revision: str | None, source_command: str | None) -> dict[str, Any]:
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
                language = _task_language(task, benchmark_id)
                normalized, scale = _normal_value(float(value), metric_name)
                metrics.append({
                    "capability": benchmark.get("capability", "unmapped"),
                    "benchmark": benchmark_id,
                    "task": task,
                    "metric": metric_name,
                    "value": round(normalized, 6),
                    "scale": scale,
                    "direction": benchmark.get("direction", "higher"),
                    "language": language,
                    "n": (raw.get("n-samples") or {}).get(task, {}).get("effective"),
                })
                break
    now = datetime.now(timezone.utc).isoformat()
    sample_counts = raw.get("n-samples") or {}
    limited_tasks = sorted(
        task for task, counts in sample_counts.items()
        if isinstance(counts, dict)
        and isinstance(counts.get("original"), (int, float))
        and isinstance(counts.get("effective"), (int, float))
        and counts["effective"] < counts["original"]
    )
    configured_limit = (raw.get("config") or {}).get("limit")
    diagnostic = configured_limit is not None or bool(limited_tasks)
    run = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed" if metrics else "failed",
        "started_at": _iso_timestamp(raw.get("date"), now),
        "finished_at": now,
        "model": {"id": model_id, "revision": model_revision, "format": "hf"},
        "provenance": {"kind": "fresh-reproduced", "source": source, "command": source_command, "harness": "lm-evaluation-harness", "harness_version": (raw.get("versions") or {}).get("lm_eval")},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "metrics": metrics,
    }
    if diagnostic:
        run["diagnostic"] = True
        detail = f"Configured per-task limit: {configured_limit}." if configured_limit is not None else f"Limited tasks: {', '.join(limited_tasks)}."
        run["limitations"] = [f"Bounded diagnostic sample; {detail} Scores are directional and must not be used as release estimates."]
    errors = validate_result(run)
    if errors:
        raise ResultError("; ".join(errors))
    return run


def ingest_lm_eval(input_path: Path, model_id: str, run_id: str, model_revision: str | None, source_command: str | None) -> dict[str, Any]:
    return _ingest_lm_eval_raw(load_json(input_path), str(input_path), model_id, run_id, model_revision, source_command)


def ingest_lm_eval_directory(input_dir: Path, model_id: str, run_id: str, model_revision: str | None, source_command: str | None) -> dict[str, Any]:
    """Merge resumable grouped lm-eval outputs into one diagnostic run."""
    if not input_dir.is_dir():
        raise ResultError(f"lm-eval input directory does not exist: {input_dir}")
    candidates: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*.json")):
        try:
            value = load_json(path)
        except ValueError:
            continue
        if isinstance(value.get("results"), dict):
            candidates.append(value)
    if not candidates:
        raise ResultError(f"No lm-eval result JSON files found under {input_dir}")
    merged: dict[str, Any] = {
        "results": {},
        "n-samples": {},
        "versions": {},
        "config": {"limit": next(((item.get("config") or {}).get("limit") for item in candidates if (item.get("config") or {}).get("limit") is not None), None)},
    }
    dates = [item.get("date") for item in candidates if item.get("date") is not None]
    if dates:
        merged["date"] = dates[0]
    for item in candidates:
        merged["results"].update(item.get("results") or {})
        merged["n-samples"].update(item.get("n-samples") or {})
        merged["versions"].update(item.get("versions") or {})
    return _ingest_lm_eval_raw(merged, str(input_dir), model_id, run_id, model_revision, source_command)


def ingest_oellm_csv(input_path: Path, model_id: str, run_id: str, model_revision: str | None, source_command: str | None, source_model: str | None = None, diagnostic: bool = False) -> dict[str, Any]:
    with input_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"model_name", "task", "n_shot", "performance", "metric_name"}
    if not rows or not required.issubset(rows[0]):
        raise ResultError(f"oellm-eval CSV must contain {', '.join(sorted(required))}")
    source_models = sorted({row["model_name"] for row in rows})
    if source_model:
        rows = [row for row in rows if row["model_name"] == source_model]
        if not rows:
            raise ResultError(f"No rows match --source-model {source_model!r}")
    elif len(source_models) > 1:
        raise ResultError(f"CSV contains multiple source models; choose one with --source-model: {', '.join(source_models)}")
    registry = benchmark_index()
    metrics = []
    for row in rows:
        benchmark_id = _task_to_benchmark(row["task"])
        benchmark = registry.get(benchmark_id)
        if benchmark is None:
            raise ResultError(f"No registered benchmark mapping for oellm-eval task {row['task']!r}")
        try:
            value, scale = _normal_value(float(row["performance"]), row["metric_name"])
        except ValueError as exc:
            raise ResultError(f"Non-numeric performance for {row['task']!r}: {row['performance']!r}") from exc
        shot = row.get("n_shot")
        slice_name = f"{shot}-shot" if shot not in (None, "", "unknown") else None
        metrics.append({
            "capability": benchmark["capability"],
            "benchmark": benchmark_id,
            "task": row["task"],
            "metric": row["metric_name"],
            "value": round(value, 6),
            "scale": scale,
            "direction": benchmark.get("direction", "higher"),
            "language": _task_language(row["task"], benchmark_id),
            "slice": slice_name,
        })
    now = datetime.now(timezone.utc).isoformat()
    run = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "completed",
        "finished_at": now,
        "model": {"id": model_id, "revision": model_revision, "format": "hf"},
        "provenance": {"kind": "fresh-reproduced", "source": str(input_path), "command": source_command, "harness": "OpenEuroLLM/oellm-eval collect"},
        "environment": {"platform": platform.platform(), "python": platform.python_version()},
        "metrics": metrics,
    }
    diagnostic = diagnostic or bool(source_command and re.search(r"(?:^|\s)--limit(?:\s|=)", source_command))
    if diagnostic:
        run["diagnostic"] = True
        run["limitations"] = ["Bounded quick-survey sample. Scores are directional and must not be used as release estimates or release-gate evidence."]
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
        if destination.exists() and json.loads(destination.read_text()) != run:
            raise ResultError(f"Run id {run['run_id']} already exists with different content; run records are immutable")
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
