# Paired SFT–DPO needle diagnostic on LUMI

This is a **completed diagnostic, not a long-context benchmark or release gate**. The
paired SFT and DPO checkpoints used identical tokenized prompts. At 2K, 8K, and
16K tokens, neither checkpoint retrieved the needle in any of nine cases. Both
retrieved it in a separate 512-token control. Because the longer test has a
zero-score floor for both models, it cannot establish whether DPO retained or
degraded long-context capability.

## What the models were asked

Each input was one chat **user message**: an archive preface, numbered filler
entries, one fact of the form `Archive note: the access code for RECORD is
CODE.`, more entries, then `What is the access code for RECORD? Reply with only
the code.` The complete, verbatim user messages for the [512-token passing
control](examples/dpo-needle-512-depth50-user-message.txt) and the [2,048-token
failing case at 50% depth](examples/dpo-needle-2048-depth50-user-message.txt)
are included in this repository. They were reconstructed deterministically
from the pinned harness and saved case metadata, because the raw JSON recorded
prompt hashes rather than full prompt text. These are the exact message *contents*;
the shared tokenizer then applied its chat template and assistant-generation
prefix. Both checkpoints received the same resulting token IDs for each case.
The linked text files have a conventional terminal file newline; the actual
user messages ended immediately after the final period in the question.

Here is the **entire 512-token control user message** (502 tokens after chat
formatting):

```text
Archive volume 240891. Read the archive before answering.
Entry 00000. A coastal survey recorded tide tables, harbor maintenance, and ferry timetables.
Entry 00001. The museum catalog describes ceramics, maps, paintings, and conservation notes.
Entry 00002. The agricultural register discusses soil samples, orchards, rainfall, and seed varieties.
Entry 00003. The library ledger tracks book repairs, donations, shelving, and reading-room hours.
Entry 00004. A transport bulletin summarizes roadworks, bicycle lanes, and station accessibility.
Entry 00005. The botanical field notes describe moss, oak trees, wildflowers, and greenhouse care.
Entry 00006. The astronomy newsletter lists public lectures, sky conditions, and telescope repairs.
Entry 00007. A municipal notice covers recycling routes, street lighting, and park maintenance.
Archive note: the access code for record-c0cd3734 is NIAH-9437B276DEDB.
Entry 00008. The music program announces rehearsals, venue changes, and instrument storage.
Entry 00009. The town archive lists routine reports about bridges, libraries, gardens, and meeting schedules.
Entry 00010. A coastal survey recorded tide tables, harbor maintenance, and ferry timetables.
Entry 00011. The museum catalog describes ceramics, maps, paintings, and conservation notes.
Entry 00012. The agricultural register discusses soil samples, orchards, rainfall, and seed varieties.
Entry 00013. The library ledger tracks book repairs, donations, shelving, and reading-room hours.
Entry 00014. A transport bulletin summarizes roadworks, bicycle lanes, and station accessibility.
Entry 00015. The botanical field notes describe moss, oak trees, wildflowers, and greenhouse care.
Entry 00016. The astronomy newsletter lists public lectures, sky conditions, and telescope repairs.

What is the access code for record-c0cd3734? Reply with only the code.
```

SFT and DPO each returned exactly
`NIAH-9437B276DEDB<end_of_turn>` on that message.

For the 2,048-token example, the fact and final question were exactly:

```text
Archive note: the access code for record-bc46ae4a is NIAH-C1C7D0A14440.
...
What is the access code for record-bc46ae4a? Reply with only the code.
```

The `...` above is an explanatory omission of intervening archive entries,
**not** part of the actual prompt; use the linked file for the complete message
(apart from its terminal file newline).
SFT and DPO each returned exactly `<end_of_turn>` on this 2,048-token message:
neither output contained the expected `NIAH-C1C7D0A14440`.
The model outputs below are the raw generated text, including any
`<end_of_turn>` marker. `\n × 48` means exactly 48 newline characters, not the
literal characters `\` and `n`. The 512-token row is a separate pilot control;
the other nine rows are the complete grid. “Actual” is the tokenized prompt
length, including chat formatting. Every case used seed `1`.

## SFT checkpoint results

[`Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft`](https://huggingface.co/Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft)
at revision `85bf18fb4f0bee6ac6270f06b1d1c6b3be200f31` retrieved the
code in the 512-token control (1/1), but in none of the 2K, 8K, or 16K cases
(0/3 at each length; 0/9 overall in the grid).

| Ceiling (actual tokens) | Depth | Record / expected answer | SFT raw result | Retrieved? |
| --- | ---: | --- | --- | --- |
| 512 (502; control) | 50% | `record-c0cd3734` / `NIAH-9437B276DEDB` | `NIAH-9437B276DEDB<end_of_turn>` | Yes |
| 2,048 (2,035) | 10% | `record-03821669` / `NIAH-0E308D157B8B` | `<end_of_turn>` | No |
| 2,048 (2,031) | 50% | `record-bc46ae4a` / `NIAH-C1C7D0A14440` | `<end_of_turn>` | No |
| 2,048 (2,034) | 90% | `record-596242e6` / `NIAH-8044ED98B5D7` | `<end_of_turn>` | No |
| 8,192 (8,181) | 10% | `record-086d097c` / `NIAH-AEA9524DE948` | `\n × 48` | No |
| 8,192 (8,179) | 50% | `record-92bc16a8` / `NIAH-C00AFDC57E90` | `\n × 48` | No |
| 8,192 (8,180) | 90% | `record-2c2bb6b5` / `NIAH-890A5604EBB1` | `\n × 48` | No |
| 16,384 (16,367) | 10% | `record-04bb6db8` / `NIAH-8CC99D11741D` | `\n × 48` | No |
| 16,384 (16,370) | 50% | `record-2194c409` / `NIAH-DBA0334BB7D2` | `Entry 6666666666666666666666666666666666666666666666` | No |
| 16,384 (16,372) | 90% | `record-3179649f` / `NIAH-5285B027C313` | `\n × 48` | No |

For example, on the linked 2,048-token prompt, SFT's **entire response** was
`<end_of_turn>`—it did not produce an access code.

## DPO checkpoint results

[`birgermoell/oellm-9b-threeway-dpo-full-v1`](https://huggingface.co/birgermoell/oellm-9b-threeway-dpo-full-v1)
at revision `df4ca88030163946040f34b7f238503707906a97` also retrieved the
code in the 512-token control (1/1), but in none of the 2K, 8K, or 16K cases
(0/3 at each length; 0/9 overall in the grid).

| Ceiling (actual tokens) | Depth | Record / expected answer | DPO raw result | Retrieved? |
| --- | ---: | --- | --- | --- |
| 512 (502; control) | 50% | `record-c0cd3734` / `NIAH-9437B276DEDB` | `NIAH-9437B276DEDB<end_of_turn>` | Yes |
| 2,048 (2,035) | 10% | `record-03821669` / `NIAH-0E308D157B8B` | `<end_of_turn>` | No |
| 2,048 (2,031) | 50% | `record-bc46ae4a` / `NIAH-C1C7D0A14440` | `<end_of_turn>` | No |
| 2,048 (2,034) | 90% | `record-596242e6` / `NIAH-8044ED98B5D7` | `<end_of_turn>` | No |
| 8,192 (8,181) | 10% | `record-086d097c` / `NIAH-AEA9524DE948` | `\n × 48` | No |
| 8,192 (8,179) | 50% | `record-92bc16a8` / `NIAH-C00AFDC57E90` | `\n × 48` | No |
| 8,192 (8,180) | 90% | `record-2c2bb6b5` / `NIAH-890A5604EBB1` | `\n × 48` | No |
| 16,384 (16,367) | 10% | `record-04bb6db8` / `NIAH-8CC99D11741D` | `\n\n\n<end_of_turn>` | No |
| 16,384 (16,370) | 50% | `record-2194c409` / `NIAH-DBA0334BB7D2` | `Entry 6666666666666666666666666666666666666666666666` | No |
| 16,384 (16,372) | 90% | `record-3179649f` / `NIAH-5285B027C313` | `\n × 48` | No |

On that *same* linked 2,048-token prompt, DPO's **entire response** was also
`<end_of_turn>`. At 16K/10% depth its result differed from SFT's, but neither
contained the requested code. A shared zero-score floor is not evidence that
their long-context abilities are equal.

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

### Follow-up prompt design

This run called the target an “access code.” A more conventional passkey probe
would use a single explicit statement such as `The passkey for record X is Y`
and ask `What is the passkey for record X?` The repository's
[existing NIAH generator](../src/post_training_eval/niah.py) uses that format,
and [RULER's passkey-retrieval task](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/ruler/README.md)
is a useful external reference. That is a **proposed new test**, not a
reinterpretation of the results above. Changing “access code” to “passkey”
alone would not establish why both models produced empty or repetitive answers;
the next run should first validate generation/template behavior with a short
control and a passing 2K SFT case, then use identical prompts for both models.
