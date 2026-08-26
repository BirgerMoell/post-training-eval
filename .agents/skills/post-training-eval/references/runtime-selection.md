# Runtime selection

Read this reference when choosing hardware or constructing harness environments.

## Choose a runtime

| Runtime | Prefer for | Avoid claiming |
|---|---|---|
| Apple Silicon, 48 GB unified memory | Short-context 9B smoke tests, selected quick tasks, local debugging | Full 128K/256K BF16 validation or exact CUDA parity |
| Single 23–24 GB CUDA GPU | One 9B BF16 model with short/medium contexts and batch size one | Long advertised context without a measured memory plan |
| Two small CUDA GPUs | One model per GPU for comparisons; model parallel only when the harness supports it explicitly | That aggregate VRAM is automatically usable by every runner |
| LUMI or another cluster | Core/release suites, large models, long context, repeatable scheduled runs | Interactive workstation assumptions or unreviewed conversions |
| OpenAI-compatible endpoint | Holdouts, tool/RAG integration, served Megatron checkpoints | Weight-level reproducibility without a pinned server build |

Quantized exploratory runs can answer engineering questions but are not protocol-matched with BF16 baselines unless both sides use the same quantization and serving stack.

## Check capacity

Record these before model download and again after loading:

- filesystem and inode availability for the repository, Hugging Face cache, temporary directory, and result directory;
- system RAM and swap;
- GPU model, driver, visible device ids, free VRAM, and other active processes;
- checkpoint weight size and whether multiple revisions share blobs;
- expected context length, batch size, precision, and generation limit.

Keep a free-disk floor large enough for raw details and temporary downloads. Do not remove shared caches to create room without explicit approval.

## Keep harnesses isolated

Use separate environments when the official task stack requires incompatible dependency versions:

1. lm-evaluation-harness environment for the largest compatible task batches;
2. Lighteval environment pinned to the reviewed revision;
3. Evalchemy environment pinned to the OpenEuroLLM-compatible fork or revision.

At the repository’s currently documented Lighteval revision, use `xxhash<4` (`xxhash==3.6.0` is smoke-tested). Re-check the repository README before applying this pin because upstream compatibility can change.

Run each harness help command and a minimal task before committing hours of GPU time. Record every package version and upstream commit in the run provenance or attached environment manifest.

## Preserve checkpoint integrity

- Evaluate an immutable Hugging Face snapshot path when possible.
- Keep tokenizer shims or compatibility exports in a separate directory.
- Hash or record the changed compatibility files.
- Convert Megatron only with an explicit, reviewed argument array in `configs/megatron*.json`.
- Validate the converted HF config, tokenizer, and safetensors before evaluation.
