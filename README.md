# OpenEuroLLM post-training evaluation

A capability-driven evaluation control plane for Hugging Face and Megatron checkpoints. It turns the post-training playbook into executable profiles, normalized evidence, regression gates, and a public model-by-capability scoreboard.

**[Open the live evaluation dashboard →](https://birgermoell.github.io/post-training-eval/)**

**[Install or inspect the AI-agent skill →](https://birgermoell.github.io/post-training-eval/skill.html)**

This repository deliberately separates three kinds of evidence:

- **Fresh reproduced** — created by a runner from a pinned checkpoint revision and retained raw artifacts.
- **Imported published** — copied from a model card or external leaderboard with a source URL and protocol.
- **Compatibility check** — proves a checkpoint can be resolved and has a coherent config/tokenizer/weight layout; it is not a quality score.

Missing evidence is displayed as missing. It is never converted to zero or averaged into a capability score.

## AI-agent skill

The repository includes one canonical, portable [`post-training-eval` skill](.agents/skills/post-training-eval/SKILL.md). It teaches Codex or Claude Code how to select hardware, run preflight checks, execute and resume the evaluation profiles, preserve provenance, compare checkpoints, and publish completed evidence.

- **Repository use:** Clone the repository. Codex discovers `.agents/skills/post-training-eval`; Claude Code discovers the project wrapper in `.claude/skills/post-training-eval`.
- **Personal use:** Run `python3 scripts/install_skill.py --agent both`. The default creates non-destructive symlinks in the current personal skill locations. Add `--mode copy` for a standalone copy.
- **Any capable agent:** Point it to the [canonical raw `SKILL.md`](https://raw.githubusercontent.com/BirgerMoell/post-training-eval/main/.agents/skills/post-training-eval/SKILL.md) and ask it to follow the file completely.

Preview personal installation without changing anything:

```bash
python3 scripts/install_skill.py --agent both --dry-run
```

The installer refuses to overwrite an existing skill. Repository-scoped discovery requires no installer.

## What is covered

The nine capability folders are the source of truth: Safety, Multilingual, Instruction & Chat, Reasoning & Knowledge, Grounding & RAG, Tools & Agents, Long Context, Coding, and Efficiency & Release. Each `suite.json` records benchmark, metric, direction, target, runner, implementation status, and one or more labeled `sources`. The first source is the primary click-through location. Direct dataset pages are preferred; composite, generated, leaderboard, or internal evaluations link to their exact manifest, runner, or protocol instead.

Every registered evaluation has a numeric release floor under [`program-floor-v1`](docs/TARGETS.md). The dashboard shows both target status and the exact direction-aware gap for every measured primary metric. These are versioned program targets and should be recalibrated only through review after a protocol-matched baseline sweep.

The executable core reuses [OpenEuroLLM/oellm-eval](https://github.com/OpenEuroLLM/oellm-eval) for SLURM scheduling, lm-evaluation-harness, lighteval, and Evalchemy. The repository adds checkpoint preparation, explicit evaluation profiles, a deterministic NIAH runner, an OpenAI-compatible multilingual holdout runner, normalized result ingestion, and GitHub Pages publication.

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .

pteval inspect --model hf://birgermoell/oellm-9b-256k-sft
pteval plan \
  --model hf://birgermoell/oellm-9b-256k-sft \
  --profile smoke \
  --output runs/sft-smoke-plan.json
pteval run --plan runs/sft-smoke-plan.json
```

The last command is a dry run. Add `--execute` only on a configured machine or cluster.

## Quick capability survey

Use `--quick` when you want a broad first impression before paying for full evaluation:

```bash
pteval plan \
  --model /immutable/path/to/hf-checkpoint \
  --quick \
  --venv-path /path/to/oellm-eval-venv \
  --output runs/model-quick-plan.json

pteval run --plan runs/model-quick-plan.json
pteval run --plan runs/model-quick-plan.json --execute
```

Quick mode expands the official `oellm-eval` `all` group and evaluates up to eight examples **per task**, rather than taking eight examples from the suite as a whole. This gives every registered multilingual, knowledge, reasoning, instruction, and coding task a small signal. It also plans one NIAH probe at 4K, 32K, 128K, and 262K, plus two deterministic examples from every multilingual holdout bucket when the model is available through an OpenAI-compatible endpoint.

Override the per-task cap with `--limit 4` or `--limit 16`. The default is intended as a broad survey, not a five-minute smoke test; the expanded `all` group contains many language-task combinations. Missing external adapters, gated evaluations, deployment measurements, and leaderboard submissions remain visible in the plan.

After collecting quick `oellm-eval` output, preserve the diagnostic label when normalizing it:

```bash
oellm-eval collect --results_dir oellm-output/QUICK_RUN --output_csv runs/quick.csv
pteval ingest-oellm-csv \
  --input runs/quick.csv \
  --source-model /immutable/path/to/hf-checkpoint \
  --model-id owner/model \
  --model-revision COMMIT_SHA \
  --run-id model-quick-YYYY-MM-DD \
  --source-command "oellm-eval schedule --task_groups all --limit 8" \
  --diagnostic \
  --output runs/model-quick.json
```

The dashboard displays these results with a yellow **Diagnostic** badge. Quick results are useful for triage and parent comparisons, but cannot satisfy release gates.

### Run quick mode on a GPU workstation

`oellm-eval` includes 438 task/language rows with several few-shot protocols and three mutually incompatible harness environments. `local-quick` keeps the official task registry but batches compatible rows so a 9B checkpoint is not reloaded hundreds of times. It is resumable, fixes batch size at one, uses the native chat template, and stops before free disk crosses the configured floor.

For Evalchemy's custom chat benchmarks, `local-quick` uses a diagnostic adapter because upstream `--limit` only bounds lm-eval-style tasks. The adapter applies the requested question cap to chat-task loaders, limits repeated benchmarks to one pass, and records the policy in `manifest.json`. It caps GPQA Diamond and HumanEval generations at 1,024 tokens and other registered Evalchemy quick tasks at 2,048 tokens. HumanEval uses Evalchemy's native debug subset (two examples per available language) because it loads examples inline, and retries the transient `filelock` fork race around functional tests. Every Evalchemy task runs in its own invocation without `--log_samples`, so its aggregate result is saved independently and the manifest is updated atomically before the next task starts. Gated or otherwise unavailable data remains recorded as missing evidence while later tasks continue, and the final manifest is marked `completed_with_failures`. These bounded settings make the run useful for early model comparison, but they are not protocol-compatible with full benchmark scores or release gates.

Use `--fast` for a much shorter triage pass before `quick`: it evaluates two examples per task by default, caps Evalchemy generations at 384 tokens (128 for GPQA Diamond), and limits upstream dataset preprocessing to four workers. This prevents LiveCodeBench from forking once per host CPU and exhausting RAM. Fast scores have high sampling uncertainty and short-output truncation risk; compare them only with another run using the same checkpoint-independent fast protocol.

Before either profile, `pteval canary` loads the exact model and tokenizer, renders two small chat prompts, and records whether greedy and sampled generation stop on EOS or hit the token limit. This catches tokenizer/chat-template errors and repetition loops without loading every benchmark:

```bash
pteval canary \
  --model /immutable/hf/snapshot \
  --tokenizer /path/to/documented/tokenizer-view \
  --max-new-tokens 128 \
  --output runs/model-canary.json
```

```bash
PYTHONPATH=/path/to/oellm-eval pteval local-quick \
  --model /immutable/hf/snapshot \
  --output-dir runs/model-quick \
  --lm-python /envs/lm-eval/bin/python \
  --include-path /path/to/oellm-eval/oellm/resources/custom_lm_eval_tasks \
  --lighteval-bin /envs/lighteval/bin/lighteval \
  --evalchemy-python /envs/evalchemy/bin/python \
  --evalchemy-dir /path/to/evalchemy \
  --fast --gpu 0 --min-free-gb 20
```

Remove `--fast` (or set `--limit 8`) for the broader quick survey. Use `--suite lm-eval-harness` while validating one runtime. Repeating the same command resumes completed batches from `manifest.json`; failed or interrupted batches are rerun. Two models can run concurrently on separate GPUs by using distinct output directories and `--gpu 0` / `--gpu 1`.

Once a fast run is terminal, normalize its manifest and saved aggregate task results for the dashboard. Raw benchmark examples remain outside the repository; the published record contains only scores, counts, runtime, protocol, and task availability:

```bash
pteval ingest-local-quick \
  --input-dir runs/model-fast \
  --model-id owner/model \
  --model-revision COMMIT_SHA \
  --run-id model-fast-YYYY-MM-DD \
  --accelerator "NVIDIA L4" \
  --source-command "pteval local-quick ... --fast" \
  --output runs/model-fast.json
```

Fast records appear in their own dashboard matrix and are excluded from the release-oriented aggregate capability index and target pass counts.

At the currently pinned Lighteval revision, keep `xxhash<4` in its isolated tool environment; `xxhash` 4.x rejects the string hashing call used while saving per-example details. The runner was smoke-tested with `xxhash==3.6.0`.

## Run on LUMI with `oellm-eval`

Install the OpenEuroLLM scheduler in the login-node environment and point `HF_HOME` at the shared cache:

```bash
uv tool install -p 3.12 git+https://github.com/OpenEuroLLM/oellm-eval.git
export HF_HOME=/scratch/project_462000963/cache/huggingface

pteval plan \
  --model birgermoell/oellm-9b-256k-reasoning-v1 \
  --profile core \
  --output runs/reasoning-core-plan.json
pteval run --plan runs/reasoning-core-plan.json --execute
```

Use `--limit 16` only for stack validation. Limited results are labeled diagnostic and must not be used for release comparisons.

## Hugging Face and Megatron checkpoints

Accepted model references:

| Reference | Meaning |
|---|---|
| `hf://owner/model@revision` | Hugging Face Hub checkpoint, optionally pinned (direct lm-eval); use a downloaded snapshot for oellm-eval task groups |
| `/absolute/path/to/hf` | Local Transformers checkpoint |
| `megatron:///absolute/path` | Distributed Megatron checkpoint requiring preparation |
| `openai://https://host/v1#served-model` | HF or Megatron model behind an OpenAI-compatible server |

Megatron tensor layouts vary with model architecture and training stack, so conversion is never guessed. Copy `configs/megatron.example.json`, pin a reviewed converter command, preview it, then execute:

```bash
pteval prepare-megatron --config configs/megatron.local.json
pteval prepare-megatron --config configs/megatron.local.json --execute
```

The converter is passed an argument array without a shell. After conversion, the output must pass the same HF config/tokenizer/safetensors checks as a native HF checkpoint. A Megatron checkpoint may instead be served directly and evaluated through the endpoint reference.

## Profiles

- `smoke` — checkpoint, 16-example reasoning/instruction and three-language multilingual stack check, 4K NIAH. Diagnostic only.
- `quick` / `--quick` — every registered `oellm-eval` task with a small per-task cap, four context lengths, and stratified endpoint holdouts. Diagnostic only.
- `core` — the per-checkpoint development suite and retention gate.
- `release` — full multilingual coverage plus required external safety, arena, agent, long-context, and efficiency evidence.

External tasks remain explicit blocking rows in a release plan. They do not silently disappear when a package, judge, gated dataset, or leaderboard submission is unavailable.

## Normalize and publish results

lm-evaluation-harness results are converted to the repository schema:

```bash
pteval ingest-lm-eval \
  --input oellm-output/RUN/results/results_*.json \
  --model-id birgermoell/oellm-9b-256k-sft \
  --model-revision 08359ad61333263c067edaf290067fea5b103d34 \
  --run-id 2026-08-26-sft-core \
  --source-command "oellm-eval schedule ..." \
  --output runs/2026-08-26-sft-core.json

pteval publish --run runs/2026-08-26-sft-core.json
```

For a mixed `oellm-eval` run (lm-eval, lighteval, and Evalchemy), collect first and ingest the unified CSV:

```bash
oellm-eval collect --results_dir oellm-output/RUN --output_csv runs/collected.csv
pteval ingest-oellm-csv \
  --input runs/collected.csv \
  --source-model /immutable/checkpoint/path \
  --model-id owner/model \
  --model-revision COMMIT_SHA \
  --run-id 2026-08-26-model-core \
  --output runs/2026-08-26-model-core.json
```

`publish` validates the run, copies it to `results/runs/`, and rebuilds `docs/data/index.json`. Use `--push` in a clean authenticated clone to commit and push the result; the Pages workflow then deploys the updated dashboard.

Compare protocol-matched runs and fail the development gate when a common slice regresses by more than two points:

```bash
pteval gate \
  --baseline results/runs/baseline.json \
  --candidate results/runs/candidate.json \
  --max-regression 2 \
  --output runs/gate.json
```

Target misses are reported separately from parent regressions. A release decision additionally requires all non-compensatory safety and required profile evidence.

## Result contract

Every metric carries:

- checkpoint id and immutable revision;
- evidence type, harness, protocol/source, and command;
- capability, benchmark, metric, direction, value, and scale;
- language/slice and sample count when applicable;
- raw artifact references and limitations.

Comparisons are valid only when protocol fields match. Language reporting uses macro, p10, minimum, and tier gap in addition to the macro average. Safety is non-compensatory, and a regression over two points on a priority slice cannot be hidden by a better aggregate.

For chat/instruction generation the direct lm-eval steps explicitly pass `--apply_chat_template`. Current `oellm-eval` task groups are used for their multilingual/custom-task registry; before comparing an instruction-tuned checkpoint with a completion baseline, confirm that the prompt protocol is identical. The plan and result must record this choice.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). A contribution can add a benchmark adapter, register a new dataset/eval, or publish a run. Do not commit benchmark test examples, API keys, private prompts, model weights, or unredacted red-team transcripts.
