---
name: post-training-eval
description: Run, resume, normalize, compare, and publish capability-driven post-training evaluations for Hugging Face, Megatron, or OpenAI-compatible model checkpoints with the OpenEuroLLM post-training-eval control plane. Use for model smoke tests, bounded all-task surveys, full release evaluations, multilingual capability analysis, checkpoint comparisons, evaluation troubleshooting, hardware selection, provenance capture, regression gates, or dashboard publication.
---

# Post-training evaluation

Use the repository as the evaluation control plane. Preserve its distinction between fresh reproduced results, imported published evidence, and checkpoint compatibility checks. Never turn missing evidence into a zero or a model-quality claim.

## Locate the control plane

1. Search the current directory and its parents for `pyproject.toml` with project name `openeurollm-post-training-eval`.
2. If it is absent, clone `https://github.com/BirgerMoell/post-training-eval` into a user-approved workspace. Do not clone into an unrelated repository.
3. Treat the repository `README.md`, `profiles/*.json`, `capabilities/*/suite.json`, and CLI help as the current source of truth. Do not rely on commands remembered from an older skill version.
4. Resolve all repository-relative paths from the discovered repository root, even when this skill is installed globally.

## Select the smallest honest evaluation

- Use `smoke` to prove checkpoint and harness compatibility. Treat its scores as diagnostic.
- Use `local-quick --fast` for two-example-per-task triage with short generation caps; use it to find broken protocols and obvious regressions, not to estimate stable benchmark scores.
- Use `local-quick --sweep` plus the sweep profile's endpoint holdouts and NIAH steps when the user needs a very small signal across all nine capability contracts. The standard local step selects eight exact registry tasks; the complete sweep contains 19 probes and remains diagnostic.
- Use `quick` or `local-quick --limit 8` for a broad first comparison across every registered operational task. Explain that it is bounded per task and cannot satisfy a release gate.
- Use `core` for repeatable checkpoint-development comparisons and parent-retention gates.
- Use `release` only when required external, safety, multilingual, long-context, agent, and efficiency evidence can be completed or explicitly reported as missing.
- Use `endpoint-run` for multilingual holdouts against an OpenAI-compatible deployment.
- Use `prepare-megatron` only with a reviewed converter configuration. Never guess a Megatron tensor-layout conversion.

Default to `quick` when the request is to “get a feel” for a model and the user has not requested a release decision. Preserve any explicitly requested profile or protocol.

## Run the mandatory preflight

Before downloading weights or launching inference:

1. Inspect git status and preserve unrelated changes.
2. Check free disk, inodes, RAM, accelerators, per-device VRAM, active GPU processes, and the intended cache location.
3. Resolve every Hugging Face model to an immutable revision and record the exact snapshot. Never compare floating `main` revisions.
4. Run `pteval inspect --model MODEL_REFERENCE` before scheduling quality evaluation.
5. Run `pteval canary --model IMMUTABLE_SNAPSHOT --tokenizer TOKENIZER_PATH --output RUN_CANARY.json` when the tokenizer requires a compatibility view or a checkpoint may loop. Treat length-stopped canaries as a protocol/model warning, not a benchmark score.
6. Estimate checkpoint, cache, temporary artifact, and long-context KV-cache requirements. Set `--min-free-gb` to a meaningful floor on shared or nearly full hosts.
7. Use one dedicated run directory per model and protocol. Never reuse a run directory for a different checkpoint revision, profile, or example limit.
8. Avoid deleting shared caches. Stop and report the constraint if the run cannot fit safely.

Read [runtime selection](references/runtime-selection.md) when choosing between a Mac, GPU workstation, LUMI, or an endpoint, or when creating harness environments.

## Set up reproducibly

Create an isolated Python environment and install the repository editable. Use the upstream `OpenEuroLLM/oellm-eval` registry for official task expansion. Pin its commit and record harness versions.

Use separate environments for lm-evaluation-harness, Lighteval, and Evalchemy when their dependency constraints conflict. Do not “fix” incompatibilities by silently changing benchmark packages or checkpoint files. If a tokenizer compatibility view is required, place it outside the immutable checkpoint and document the change.

Preview work before execution:

```bash
pteval inspect --model hf://owner/model@REVISION
pteval plan --model hf://owner/model@REVISION --quick --output runs/model-quick-plan.json
pteval run --plan runs/model-quick-plan.json
```

Only add `--execute` after the plan, hardware, cache, and credentials have been checked.

## Execute and resume

For a CUDA workstation with a local immutable Hugging Face snapshot, prefer `pteval local-quick`. Supply the exact lm-eval custom-task path and the isolated harness executables. Keep `--limit`, batch size, chat-template choice, precision, GPU assignment, and free-disk floor visible in the recorded command.

For LUMI, use the repository SLURM guidance and `oellm-eval` scheduler rather than translating workstation commands ad hoc. For a Mac, limit the first pass to short-context smoke or quick tasks that fit unified memory; do not imply that a 128K or 256K advertised context has been validated by short examples.

Monitor long runs without starting duplicates:

- Inspect `manifest.json`, logs, process state, GPU utilization, memory, and disk periodically.
- Re-run the identical `local-quick` command to resume completed batches.
- Preserve failed-batch logs and raw result files.
- Distinguish slow generation from a stalled process using progress, utilization, and log timestamps.
- Report partial progress as partial progress; do not publish it as a completed model scorecard.

## Normalize and compare

Normalize raw harness output into the repository run schema. Use `ingest-lm-eval-dir` for grouped resumable lm-eval output, `ingest-oellm-csv` for a unified OpenEuroLLM collector CSV, and `ingest-local-quick` for a terminal workstation manifest with per-task aggregate saves. Use `ingest-endpoint`, `ingest-niah`, and `ingest-inspect` for the other sweep artifacts; these importers keep prompts, answers, secrets, and licensed examples out of published records. Mark every bounded run `diagnostic`. Fast and sweep records must not enter the main capability aggregate or release target pass counts.

Require every measurement to retain model revision, harness and task version, prompt/chat protocol, few-shot count, decoding settings, limit/sample count, language or slice, raw artifact location, and limitations. Refuse comparisons when material protocol fields differ.

Use `pteval gate` only on protocol-matched parent and candidate runs. Keep safety failures and priority-language regressions non-compensatory; do not hide them inside an average.

Read [result lifecycle](references/result-lifecycle.md) before normalizing mixed-harness output, comparing models, or publishing results.

## Publish deliberately

Validate the normalized run, registry, tests, and generated site before publication. Use `pteval publish --run ...` to copy a completed run into `results/runs/` and rebuild `docs/data/index.json`.

Push or open a pull request only when the user explicitly authorizes the external change. Never commit model weights, API keys, private prompts, licensed benchmark examples, or unredacted red-team transcripts. Verify the GitHub Pages deployment and make clear whether the dashboard contains completed, diagnostic, imported, or compatibility-only evidence.

## Report the outcome

Lead with whether the run completed and whether the dashboard was updated. Include:

- exact model id and revision;
- profile, task count, limit, harnesses, chat protocol, and hardware;
- completed, failed, skipped, and still-running work;
- disk/VRAM constraints or protocol deviations;
- result and raw-artifact paths;
- regression-gate outcome and missing release evidence;
- dashboard and repository links when publication occurred.

Never describe a diagnostic run, checkpoint load, or compatibility check as a full evaluation.
