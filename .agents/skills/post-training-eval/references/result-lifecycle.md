# Result lifecycle

Read this reference for mixed-harness normalization, comparisons, release decisions, or publication.

## Evidence classes

- `fresh-reproduced`: A completed runner execution against a pinned revision with retained raw artifacts.
- `imported-published`: A score copied from a model card or leaderboard with an exact source URL and protocol.
- `compatibility-check`: Evidence that the checkpoint resolves and loads coherently. It is not a quality score.

Set `diagnostic: true` for smoke, quick, `--limit`, or otherwise bounded results. Diagnostic evidence may guide triage but cannot satisfy release requirements.

## Normalize

Use the narrowest matching importer:

```bash
pteval ingest-lm-eval-dir \
  --input-dir RAW_RESULTS \
  --model-id owner/model \
  --model-revision COMMIT_SHA \
  --run-id model-quick-YYYY-MM-DD \
  --source-command "EXACT COMMAND" \
  --output runs/model-quick.json
```

Use `ingest-oellm-csv` after `oellm-eval collect` when lm-eval, Lighteval, and Evalchemy have been collected into one CSV. Select the exact source model if the CSV includes several models. Retain the CSV and raw harness directories outside the published summary when they are too large or contain protected examples.

## Validate comparisons

Compare only when these fields match materially:

- benchmark/task and dataset revision;
- prompt, chat template, system prompt, and answer extraction;
- few-shot count and sampling policy;
- decoding and generation limits;
- precision, quantization, and harness version when they can change results;
- language/slice and sample count or full-test status.

Report multilingual macro, p10, minimum, and tier gap. Show priority-language regressions separately. Treat safety gates as non-compensatory.

## Publish

Before publication:

1. Confirm all scheduled batches reached a terminal state and investigate failures.
2. Validate the normalized run against `schemas/run.schema.json`.
3. Run `pteval validate` and the repository unit tests.
4. Run `pteval site` and confirm only intended generated data changed.
5. Inspect the dashboard locally or after deployment.
6. Use `pteval publish --run RUN.json` only for completed evidence.
7. Use `--push` only with explicit authorization and a clean authenticated clone.

After deployment, verify the Pages URL returns successfully and that the model, immutable revision, evidence badge, limitations, task slices, and scores are visible. State explicitly when a run remains diagnostic or when required capability evidence is missing.
