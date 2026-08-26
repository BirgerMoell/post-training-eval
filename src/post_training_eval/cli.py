from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .checkpoints import CheckpointError, inspect_checkpoint, parse_checkpoint, prepare_megatron
from .planner import build_plan, write_plan
from .registry import repository_root, validate_registry
from .results import ResultError, git_publish, ingest_lm_eval, ingest_lm_eval_directory, ingest_oellm_csv, publish_run, validate_result, write_site_data
from .holdouts import run_endpoint
from .gates import compare_runs
from .local_quick import LocalQuickError, run_local_quick
from .registry import load_json


def _write(value: dict, output: str | None) -> None:
    rendered = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered)
    else:
        print(rendered, end="")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pteval", description="Capability-driven OpenEuroLLM post-training evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_p = sub.add_parser("inspect", help="Validate an HF, local HF, Megatron, or endpoint checkpoint reference")
    inspect_p.add_argument("--model", required=True)
    inspect_p.add_argument("--output")
    prepare_p = sub.add_parser("prepare-megatron", help="Run a declared Megatron-to-HF converter")
    prepare_p.add_argument("--config", required=True)
    prepare_p.add_argument("--execute", action="store_true")
    prepare_p.add_argument("--output")
    plan_p = sub.add_parser("plan", help="Resolve a capability profile to concrete runner commands")
    plan_p.add_argument("--model", required=True)
    profile_selection = plan_p.add_mutually_exclusive_group()
    profile_selection.add_argument("--profile", choices=("smoke", "quick", "core", "release"))
    profile_selection.add_argument("--quick", action="store_true", help="Sample every operational task for a broad, diagnostic capability survey")
    plan_p.add_argument("--limit", type=int, help="Override the quick/per-step example cap for lm-eval and oellm-eval tasks")
    plan_p.add_argument("--venv-path")
    plan_p.add_argument("--output")
    run_p = sub.add_parser("run", help="Execute runnable steps from a generated plan")
    run_p.add_argument("--plan", required=True)
    run_p.add_argument("--execute", action="store_true", help="Required safety switch; without it only prints commands")
    local_p = sub.add_parser("local-quick", help="Run the complete bounded task registry efficiently on a GPU workstation")
    local_p.add_argument("--model", required=True, help="Local immutable Hugging Face snapshot")
    local_p.add_argument("--output-dir", required=True)
    local_p.add_argument("--lm-python", required=True)
    local_p.add_argument("--include-path", required=True)
    local_p.add_argument("--lighteval-bin")
    local_p.add_argument("--evalchemy-python")
    local_p.add_argument("--evalchemy-dir")
    local_p.add_argument("--evalchemy-tokenizer")
    local_p.add_argument("--suite", action="append", choices=("lm-eval-harness", "lighteval", "evalchemy"))
    local_p.add_argument("--limit", type=int, default=8)
    local_p.add_argument("--gpu", default="0")
    local_p.add_argument("--max-tasks-per-invocation", type=int, default=32)
    local_p.add_argument("--min-free-gb", type=float, default=20)
    local_p.add_argument("--hf-home")
    endpoint_p = sub.add_parser("endpoint-run", help="Run multilingual holdouts against an OpenAI-compatible server")
    endpoint_p.add_argument("--base-url", required=True)
    endpoint_p.add_argument("--model", required=True)
    endpoint_p.add_argument("--suite", default="oellm-eu-eval-holdouts-v1")
    endpoint_p.add_argument("--data")
    endpoint_sampling = endpoint_p.add_mutually_exclusive_group()
    endpoint_sampling.add_argument("--limit", type=int)
    endpoint_sampling.add_argument("--samples-per-bucket", type=int, help="Deterministically sample this many examples from every capability bucket")
    endpoint_p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    endpoint_p.add_argument("--out", required=True)
    ingest_p = sub.add_parser("ingest-lm-eval", help="Normalize an lm-evaluation-harness results JSON")
    ingest_p.add_argument("--input", required=True)
    ingest_p.add_argument("--model-id", required=True)
    ingest_p.add_argument("--model-revision")
    ingest_p.add_argument("--run-id", required=True)
    ingest_p.add_argument("--source-command")
    ingest_p.add_argument("--output", required=True)
    ingest_dir_p = sub.add_parser("ingest-lm-eval-dir", help="Merge and normalize grouped/resumable lm-eval result files")
    ingest_dir_p.add_argument("--input-dir", required=True)
    ingest_dir_p.add_argument("--model-id", required=True)
    ingest_dir_p.add_argument("--model-revision")
    ingest_dir_p.add_argument("--run-id", required=True)
    ingest_dir_p.add_argument("--source-command")
    ingest_dir_p.add_argument("--output", required=True)
    collect_p = sub.add_parser("ingest-oellm-csv", help="Normalize the unified CSV produced by oellm-eval collect")
    collect_p.add_argument("--input", required=True)
    collect_p.add_argument("--model-id", required=True)
    collect_p.add_argument("--model-revision")
    collect_p.add_argument("--source-model", help="Exact model_name to select when the CSV contains multiple models")
    collect_p.add_argument("--run-id", required=True)
    collect_p.add_argument("--source-command")
    collect_p.add_argument("--diagnostic", action="store_true", help="Mark bounded/quick collector output as non-release evidence")
    collect_p.add_argument("--output", required=True)
    publish_p = sub.add_parser("publish", help="Validate a run, add it to the registry, rebuild Pages data, optionally push")
    publish_p.add_argument("--run", required=True)
    publish_p.add_argument("--push", action="store_true")
    gate_p = sub.add_parser("gate", help="Compare a candidate with a protocol-matched baseline and fail on regressions")
    gate_p.add_argument("--candidate", required=True)
    gate_p.add_argument("--baseline", required=True)
    gate_p.add_argument("--max-regression", type=float, default=2.0)
    gate_p.add_argument("--output")
    sub.add_parser("site", help="Rebuild the static Pages data file")
    sub.add_parser("validate", help="Validate capability manifests and all published runs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            _write(inspect_checkpoint(parse_checkpoint(args.model)), args.output)
        elif args.command == "prepare-megatron":
            _write(prepare_megatron(Path(args.config), args.execute), args.output)
        elif args.command == "plan":
            profile = "quick" if args.quick else (args.profile or "smoke")
            plan = build_plan(parse_checkpoint(args.model), profile, args.limit, args.venv_path)
            if args.output:
                write_plan(plan, Path(args.output))
            else:
                _write(plan, None)
        elif args.command == "run":
            plan = json.loads(Path(args.plan).read_text())
            if plan.get("profile", {}).get("diagnostic"):
                print("DIAGNOSTIC quick survey: bounded scores are directional only and cannot satisfy release gates.")
            for step in plan["steps"]:
                command = step.get("command")
                if not command or not step.get("runnable"):
                    print(f"SKIP {step['id']}: {step.get('blocked_by') or step.get('notes') or 'external runner'}")
                    continue
                print(f"RUN  {step['command_display']}")
                if args.execute:
                    subprocess.run(command, check=True)
            if not args.execute:
                print("Dry run only. Add --execute to launch runnable steps.")
        elif args.command == "local-quick":
            manifest = run_local_quick(
                model=args.model,
                output_dir=Path(args.output_dir),
                lm_python=args.lm_python,
                include_path=Path(args.include_path),
                lighteval_bin=args.lighteval_bin,
                evalchemy_python=args.evalchemy_python,
                evalchemy_dir=Path(args.evalchemy_dir) if args.evalchemy_dir else None,
                evalchemy_tokenizer=args.evalchemy_tokenizer,
                suites=args.suite or ("lm-eval-harness", "lighteval", "evalchemy"),
                limit=args.limit,
                gpu=args.gpu,
                max_tasks_per_invocation=args.max_tasks_per_invocation,
                min_free_gb=args.min_free_gb,
                hf_home=args.hf_home,
            )
            print(manifest)
        elif args.command == "endpoint-run":
            import os
            report = run_endpoint(args.base_url, args.model, args.suite, Path(args.out), args.data, args.limit, os.environ.get(args.api_key_env), args.samples_per_bucket)
            print(json.dumps({key: report[key] for key in ("model", "suite", "n", "overall_accuracy")}, indent=2))
        elif args.command == "ingest-lm-eval":
            run = ingest_lm_eval(Path(args.input), args.model_id, args.run_id, args.model_revision, args.source_command)
            _write(run, args.output)
        elif args.command == "ingest-lm-eval-dir":
            run = ingest_lm_eval_directory(Path(args.input_dir), args.model_id, args.run_id, args.model_revision, args.source_command)
            _write(run, args.output)
        elif args.command == "ingest-oellm-csv":
            run = ingest_oellm_csv(Path(args.input), args.model_id, args.run_id, args.model_revision, args.source_command, args.source_model, args.diagnostic)
            _write(run, args.output)
        elif args.command == "publish":
            root = repository_root()
            destination = publish_run(Path(args.run), root)
            write_site_data(root)
            run = json.loads(destination.read_text())
            if args.push:
                git_publish(root, run["run_id"])
            print(destination)
        elif args.command == "gate":
            report = compare_runs(load_json(Path(args.candidate)), load_json(Path(args.baseline)), args.max_regression)
            _write(report, args.output)
            return 1 if report["status"] == "failed" else 0
        elif args.command == "site":
            print(write_site_data())
        elif args.command == "validate":
            errors = validate_registry()
            for path in sorted((repository_root() / "results" / "runs").glob("*.json")):
                errors.extend(f"{path}: {error}" for error in validate_result(json.loads(path.read_text())))
            if errors:
                print("\n".join(errors), file=sys.stderr)
                return 1
            print("Registry and published runs are valid.")
    except (CheckpointError, LocalQuickError, ResultError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
