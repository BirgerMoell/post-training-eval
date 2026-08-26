## What changes?

- [ ] Capability/benchmark registry
- [ ] Runner or checkpoint adapter
- [ ] Evaluation result
- [ ] Documentation/dashboard

## Evidence checklist

- [ ] Checkpoint id and immutable revision are recorded.
- [ ] Dataset/task revision, prompt format, few-shot count, decoding, harness version, and sample count are recorded.
- [ ] The result is labeled `fresh-reproduced`, `imported-published`, or `compatibility-check` correctly.
- [ ] Raw artifacts are linked or content-hashed, without committing sensitive red-team content or licensed benchmark examples.
- [ ] Per-language results include macro, p10, minimum, and tier-gap reporting when applicable.
- [ ] Parent regressions and non-compensatory safety failures are visible rather than averaged away.
- [ ] `pteval validate`, unit tests, and regenerated Pages data pass.

## Review notes

Describe protocol deviations, missing coverage, failed gates, and evidence this result supersedes.

