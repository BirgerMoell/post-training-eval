#!/bin/bash
#SBATCH --job-name=pteval-smoke
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
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

: "${MODEL:?Set MODEL to a local HF checkpoint or Hub id}"
: "${TAG:?Set TAG to an immutable run label}"
: "${TASKS:=arc_challenge,ifeval}"
: "${LIMIT:=16}"
: "${RESULTS_ROOT:=$SLURM_SUBMIT_DIR/runs/raw}"
: "${CONTAINER:=/scratch/project_462000963/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif}"
: "${PYLIBS_OVERLAY:=/scratch/project_465002530/users/bmoell/pylibs-overlay}"
: "${HF_HOME:=/scratch/project_462000963/cache/huggingface}"
: "${STEP_TIMEOUT:=20m}"

RESULTS="$RESULTS_ROOT/$TAG"
mkdir -p "$RESULTS"

echo "model=$MODEL tag=$TAG tasks=$TASKS limit=$LIMIT timeout=$STEP_TIMEOUT"
set +e
timeout --signal=TERM --kill-after=30s "$STEP_TIMEOUT" srun --label singularity exec \
  -B /pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd \
  "$CONTAINER" bash -lc "
    export PYTHONPATH='$PYLIBS_OVERLAY':\${PYTHONPATH:-}
    export HF_HOME='$HF_HOME'
    export HF_DATASETS_CACHE='$HF_HOME/datasets'
    export HF_HUB_OFFLINE=1
    python -m lm_eval \
      --model hf \
      --model_args pretrained='$MODEL',dtype=bfloat16,trust_remote_code=True \
      --tasks '$TASKS' \
      --batch_size 1 \
      --apply_chat_template \
      --limit '$LIMIT' \
      --output_path '$RESULTS' \
      --log_samples
  "
status=$?
set -e
if [ "$status" -ne 0 ]; then
  if find "$RESULTS" -type f -name 'results_*.json' -print -quit | grep -q .; then
    echo "warning=runner exited $status during teardown after writing a results JSON"
  else
    echo "error=runner exited $status before writing results" >&2
    exit "$status"
  fi
fi

echo "completed=$RESULTS"
