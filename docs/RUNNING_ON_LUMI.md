# Running on LUMI

The repository profiles use the official `oellm-eval` scheduler for normal runs. `slurm/lumi_lm_eval.sbatch` is a deliberately small direct harness job for checking a checkpoint and the local software stack before scheduling the wider task groups.

## Diagnostic comparison of two checkpoints

```bash
mkdir -p logs runs/raw

sbatch --export=ALL,\
MODEL=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts/models/oellm-9b-256k-sft,\
TAG=sft-diagnostic,\
TASKS=arc_challenge\,ifeval,\
LIMIT=16 \
slurm/lumi_lm_eval.sbatch

sbatch --export=ALL,\
MODEL=/scratch/project_465002530/users/bmoell/oellm-reasoning-training/artifacts/releases/oellm-9b-256k-reasoning-v1,\
TAG=reasoning-v1-diagnostic,\
TASKS=arc_challenge\,ifeval,\
LIMIT=16 \
slurm/lumi_lm_eval.sbatch
```

The job uses one MI250X GCD, BF16, batch size 1, native chat template, greedy harness defaults, logged samples, and offline Hugging Face caches. The limited score is only a diagnostic. Use the full `core` profile for a comparison or gate.

## Air-gapped execution

Run `oellm-eval ... --download_only true` on the login node before scheduling a profile whose datasets are not already in the shared cache. Compute nodes use `HF_HUB_OFFLINE=1`; a cache miss should fail rather than fetch mutable data during evaluation.

## Raw result retention

Keep raw results outside git until reviewed. Normalize the selected `results_*.json` with `pteval ingest-lm-eval`, inspect the generated run JSON, and publish only the normalized record. Unsafe generations, personal information, and licensed/gated benchmark examples must not be committed.

