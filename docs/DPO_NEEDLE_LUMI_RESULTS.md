# Paired SFT–DPO needle diagnostic on LUMI

This is a **completed diagnostic, not a long-context benchmark or release gate**. The
paired SFT and DPO checkpoints used identical tokenized prompts. At 2K, 8K, and
16K tokens, neither checkpoint retrieved the needle in any of nine cases. Both
retrieved it in a separate 512-token control. Because the longer test has a
zero-score floor for both models, it cannot establish whether DPO retained or
degraded long-context capability.

## What failed

The prompt was a synthetic archive containing one access-code sentence amid
unrelated entries, followed by “What is the access code for [record]? Reply with
only the code.” The table describes the **raw generated text**, before removing
the end-of-turn marker. `\n × 48` means 48 newline characters, not the literal
characters `\` and `n`.

| Prompt-token ceiling | Needle depth | SFT raw output | DPO raw output | Retrieval |
| --- | ---: | --- | --- | --- |
| 512 (control) | 50% | Correct code + `<end_of_turn>` | Correct code + `<end_of_turn>` | Both pass |
| 2,048 | 10%, 50%, 90% | `<end_of_turn>` (one generated token) | `<end_of_turn>` (one generated token) | Both fail all three |
| 8,192 | 10%, 50%, 90% | `\n × 48` | `\n × 48` | Both fail all three |
| 16,384 | 10% | `\n × 48` | `\n\n\n<end_of_turn>` | Both fail |
| 16,384 | 50% | `Entry 6666666666666666666666666666666666666666666666` | Same text | Both fail |
| 16,384 | 90% | `\n × 48` | `\n × 48` | Both fail |

For example, at 2,048 tokens and 50% depth, the expected code was
`NIAH-C1C7D0A14440`; **both** models generated only `<end_of_turn>`. At 8,192
tokens and 50% depth, the expected code was `NIAH-C00AFDC57E90`; both generated
48 newlines. At 512 tokens, the expected code was `NIAH-9437B276DEDB`, and both
generated `NIAH-9437B276DEDB<end_of_turn>`.

The [pilot raw JSON](../runs/dpo-needle-pilot-lumi-v1.json) contains all six
512/2K/8K responses. The [grid raw JSON](../runs/dpo-needle-grid-lumi-v1.json)
contains all 18 responses, expected codes, prompt hashes, actual token lengths,
and generation counts. These are synthetic diagnostic artifacts, **not published
benchmark scores**. The pilot and grid both finished with `partial: false`.

## Exact comparison protocol

- SFT: [`Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft`](https://huggingface.co/Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft) at `85bf18fb4f0bee6ac6270f06b1d1c6b3be200f31`.
- DPO: [`birgermoell/oellm-9b-threeway-dpo-full-v1`](https://huggingface.co/birgermoell/oellm-9b-threeway-dpo-full-v1) at `df4ca88030163946040f34b7f238503707906a97`.
- Shared tokenizer: the repaired tokenizer packaged with the DPO checkpoint. The published SFT tokenizer lacks `tokenizer.model`; this is a comparison of the two weight sets under one tokenizer, not a comparison of their native release packaging.
- Synthetic archive prompts at approximately 512 (pilot only), 2,048, 8,192, and 16,384 tokens; needle depths 10%, 50%, and 90% in the grid; one seed (`1`). The 512-token pilot used 50% depth.
- Chat format: one user message, `add_generation_prompt=True`, `enable_thinking=False`; greedy decoding; maximum 48 generated tokens; stop on `<|im_end|>` or `<end_of_turn>`.
- LUMI `small-g`, one MI250 GPU, BF16 weights, FlashAttention 2. Slurm pilot job `22603425`; grid job `22607680`. Both completed with exit code 0.
- Implementation: [paired-needle harness](../scripts/compare_dpo_needle.py) and [LUMI batch script](../slurm/lumi_dpo_needle.sbatch). The LUMI script expects the pinned local snapshots and the container paths shown in that file.

Each grid case's SFT and DPO records were checked for matching prompt SHA-256,
expected code, and actual token length. There were nine unique prompts and 18
completed model responses. The exact per-case fields are in the grid JSON.

## Interpretation and limitations

The observed result is **0/9 retrieval for SFT and 0/9 for DPO**, not a measured
long-context tie. The short control proves that this setup can elicit the code
at 512 tokens, but the abrupt turn ending and repetition in longer prompts
suggest a prompt/template, tokenizer, decoding, or model-behavior problem that
this test does not isolate. A single seed and a synthetic archive are also too
narrow for a capability claim. The DPO run trained with a 2K maximum pair length;
this probe does not determine the effect of that choice.

A useful follow-up would first establish a passing 2K–16K control with a
protocol validated against the SFT checkpoint, then repeat the *same* prompts
and decoding against DPO, ideally across several seeds and a broader
long-context benchmark. Until then, neither DPO degradation nor retention of
long-context performance is demonstrated by these results.
