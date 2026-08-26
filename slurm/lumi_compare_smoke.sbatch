#!/bin/bash
#SBATCH --job-name=pteval-compare
#SBATCH --account=project_465002530
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -euo pipefail
module purge

: "${SFT_MODEL:?Set SFT_MODEL to the parent HF checkpoint}"
: "${REASONING_MODEL:?Set REASONING_MODEL to the candidate HF checkpoint}"
: "${TASKS:=arc_challenge,ifeval}"
: "${LIMIT:=16}"
: "${RESULTS_ROOT:=$SLURM_SUBMIT_DIR/runs/raw}"
: "${CONTAINER:=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif}"
: "${PYLIBS_OVERLAY:=/scratch/project_465002530/users/bmoell/pylibs-overlay}"
: "${HF_HOME:=/scratch/project_462000963/cache/huggingface}"

run_checkpoint() {
  local model="$1"
  local tag="$2"
  local results="$RESULTS_ROOT/$tag"
  mkdir -p "$results"
  echo "model=$model tag=$tag tasks=$TASKS limit=$LIMIT"
  srun --label singularity exec \
    -B /pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd \
    "$CONTAINER" bash -lc "
      export PYTHONPATH='$PYLIBS_OVERLAY':\${PYTHONPATH:-}
      export HF_HOME='$HF_HOME'
      export HF_DATASETS_CACHE='$HF_HOME/datasets'
      export HF_HUB_OFFLINE=1
      python -m lm_eval \
        --model hf \
        --model_args pretrained='$model',dtype=bfloat16,trust_remote_code=True \
        --tasks '$TASKS' \
        --batch_size 1 \
        --apply_chat_template \
        --limit '$LIMIT' \
        --output_path '$results' \
        --log_samples
    "
  echo "completed=$results"
}

run_checkpoint "$SFT_MODEL" "2026-08-26-sft-chat-smoke"
run_checkpoint "$REASONING_MODEL" "2026-08-26-reasoning-v1-chat-smoke"

