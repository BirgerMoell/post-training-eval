# Local paired needle test: SFT vs full DPO

This is a small diagnostic for whether the full three-source DPO checkpoint can still retrieve a fact from beyond its 2K-token training examples. It is **not** a general long-context score or a release gate.

The pinned checkpoints are:

| Label | Hugging Face revision |
| --- | --- |
| SFT | `Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft@85bf18fb4f0bee6ac6270f06b1d1c6b3be200f31` |
| DPO | `birgermoell/oellm-9b-threeway-dpo-full-v1@df4ca88030163946040f34b7f238503707906a97` |

Use the same DPO tokenizer for both sets of weights. The published SFT tokenizer lacks `tokenizer.model` and does not load in the current Transformers runtime; the DPO release includes the repaired tokenizer. Ordinary English text has matching token IDs in both published `tokenizer.json` files, but ChatML special-token handling differs. The default **chat** prompt follows the released model's chat template and is identical for both models; this is a paired weight comparison *under the repaired tokenizer*, not a comparison of each release with its native tokenizer packaging. A short SFT canary answered correctly under this tokenizer. `--prompt-format plain` avoids chat markers but is less representative of instruction use. Both model configs specify `rope_parameters.rope_theta=64000000`, and the current Transformers Qwen3 rotary embeddings yield identical inverse frequencies. The DPO config additionally has a top-level `rope_theta=10000`; this field is not used by the tested rotary implementation.

On a Mac with an Apple GPU and enough unified memory/disk space, download the immutable snapshots with `huggingface_hub.snapshot_download` and keep the returned snapshot paths. The two weight sets total about 54.6 GB (SFT 18.2 GB; DPO 36.4 GB), before caches and outputs. Do not download them if the remaining disk space would be unsafe.

```python
from huggingface_hub import snapshot_download

sft = snapshot_download(
    "Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft",
    revision="85bf18fb4f0bee6ac6270f06b1d1c6b3be200f31",
)
dpo = snapshot_download(
    "birgermoell/oellm-9b-threeway-dpo-full-v1",
    revision="df4ca88030163946040f34b7f238503707906a97",
)
print(sft, dpo)
```

From the repository root, first prepare prompts without loading weights:

```bash
python3 scripts/compare_dpo_needle.py \
  --tokenizer "$DPO_SNAPSHOT" \
  --lengths 2048,8192,16384 --depths 10,50,90 --seeds 1 \
  --prepare-only --out runs/dpo-needle-prepare.json
```

Then run the paired comparison, with `SFT_SNAPSHOT` and `DPO_SNAPSHOT` set to the downloaded immutable directories:

```bash
python3 scripts/compare_dpo_needle.py \
  --tokenizer "$DPO_SNAPSHOT" \
  --model "sft=$SFT_SNAPSHOT" --model "dpo=$DPO_SNAPSHOT" \
  --lengths 2048,8192,16384 --depths 10,50,90 --seeds 1 \
  --out runs/dpo-needle-paired.json
```

This gives nine cases per checkpoint. The secret, varied archive entries, tokenized prompt, generation settings, and depth are identical for both models. Run the same command again to resume missing cells. Use `--only-model sft` or `--only-model dpo` to run one side before the other with the same output file; the `partial` flag stays true until both sides finish. The JSON keeps raw answers and token counts. If the first pass fits, repeat with `--seeds 1,2,3` in a **new** output file and optionally add `32768` to the lengths. Compare pass rates by length and depth, not only the pooled total. A fall at 8K or 16K but not 2K is a regression signal worth following up with a broader long-context benchmark. An equal result here does not establish that long-context capability was retained generally.

Do not publish the raw JSON as a completed benchmark: it contains a tiny number of synthetic, deterministic cases. The original run trained with 2K maximum pair length, so this test probes extrapolation rather than what was directly trained.
