# Evaluation target policy

`program-floor-v1` defines minimum release targets for every benchmark in the capability registry. The values are planning floors for a competitive 9B–30B European post-trained model, not claims of frontier performance. Stretch values, where present, describe a deliberately optimistic checkpoint goal.

The benchmark entries in `capabilities/*/suite.json` are the machine-readable source of truth. Every entry declares:

- a primary metric and whether higher or lower is better;
- a numeric `target` used for the on/off-target decision;
- an optional `stretch` goal;
- a pinned runner or an explicit adapter/submission dependency.

Targets were chosen as capability-level release floors:

- established public benchmarks use a strong but plausible 9B–30B threshold;
- OpenEuroLLM holdouts use 70–90% floors depending on whether the contract is generative, tool-structured, or safety-critical;
- safety limits are non-compensatory and use low harmful-compliance or over-refusal ceilings;
- long-context retrieval uses a 95% floor, while harder synthesis/reasoning suites use lower task-specific floors;
- efficiency targets assume batch 1 on one modern 64 GB-class accelerator, a 2K-token prompt, and a 512-token decode. Hardware, engine, precision, and model size must be recorded with the result before the target is used as a release gate.

The first calibration pass used primary reference points rather than copying a single model: the [Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3) for 12B/27B reasoning, instruction, code, and multilingual ranges; the [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) for the target size class, reasoning protocol, tool-use expectations, and 128K deployment behavior; and the official [HarmBench](https://github.com/centerforaisafety/HarmBench), [XSTest](https://github.com/paul-rottger/xstest), and [BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) contracts for safety and tool metrics. Protocol differences still prevent treating those public scores as directly comparable results.

The dashboard reports a signed, direction-aware gap. Positive means the model is beyond the target; negative means it is short. For lower-is-better metrics, “inside limit” and “over limit” replace above/below language. Missing evidence has no gap and never counts as a pass.

Targets must be reviewed after the first protocol-matched baseline sweep. A target change requires a normal repository review because it changes the release contract; past result values remain immutable.
