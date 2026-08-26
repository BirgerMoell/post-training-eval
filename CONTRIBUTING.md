# Contributing

## Add or change a benchmark

1. Edit exactly one `capabilities/<capability>/suite.json`.
2. Give the benchmark a stable id, one primary metric, direction, runner, status, and canonical source URL.
3. Record the exact dataset revision and prompt/scorer protocol in the runner or result. Do not rely on a mutable dataset name alone.
4. Add it to a profile only when the command is reproducible. External work stays visible as `adapter-required`, `license-gated`, or `submission-required`.
5. Run `python3 -m post_training_eval.cli validate` and `python3 -m unittest discover -s tests -v`.

## Publish evidence

Use `pteval ingest-lm-eval` or construct a result matching `schemas/run.schema.json`. Keep raw generations outside git if they may contain unsafe or personal content, but retain their content hash and controlled storage location. A published run must not mix fresh and imported evidence.

For a full evaluation, open a pull request containing the normalized run, regenerated `docs/data/index.json`, relevant protocol notes, and links to raw artifacts. At least one reviewer should confirm the checkpoint revision, task revision, chat template, few-shot count, decoding settings, and sample count.

Run ids are immutable. Republishing byte-equivalent evidence is idempotent; changing an existing run under the same id is rejected. Corrections must use a new run id and explain which earlier evidence they supersede.

## Status vocabulary

- `operational`: a maintained command is available.
- `adapter-required`: the benchmark is selected but not yet connected here.
- `license-gated`: access/terms must be resolved before execution.
- `submission-required`: the score comes from an external service or leaderboard.
- `planned`: specification or data is not yet ready.
