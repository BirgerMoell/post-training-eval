# OpenEuroLLM post-training evaluation

A capability-driven evaluation control plane for Hugging Face and Megatron checkpoints. It turns the post-training playbook into executable profiles, normalized evidence, regression gates, and a public model-by-capability scoreboard.

**[Open the live evaluation dashboard →](https://birgermoell.github.io/post-training-eval/)**

This repository deliberately separates three kinds of evidence:

- **Fresh reproduced** — created by a runner from a pinned checkpoint revision and retained raw artifacts.
- **Imported published** — copied from a model card or external leaderboard with a source URL and protocol.
- **Compatibility check** — proves a checkpoint can be resolved and has a coherent config/tokenizer/weight layout; it is not a quality score.

Missing evidence is displayed as missing. It is never converted to zero or averaged into a capability score.

## What is covered

The nine capability folders are the source of truth: Safety, Multilingual, Instruction & Chat, Reasoning & Knowledge, Grounding & RAG, Tools & Agents, Long Context, Coding, and Efficiency & Release. Each `suite.json` records benchmark, metric, direction, target, runner, and implementation status.

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
