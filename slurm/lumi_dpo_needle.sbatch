#!/bin/bash
#SBATCH --job-name=pteval-dpo-needle
#SBATCH --account=project_465002530
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/flash/project_465002530/users/bmoell/helpsteer3-dpo/needle-eval/logs/%j.out
#SBATCH --error=/flash/project_465002530/users/bmoell/helpsteer3-dpo/needle-eval/logs/%j.err

set -euo pipefail
module purge

ROOT=/flash/project_465002530/users/bmoell/helpsteer3-dpo/needle-eval
SFT=/scratch/project_465002530/users/bmoell/helpsteer3-dpo/artifacts/model
DPO=/flash/project_465002530/users/bmoell/helpsteer3-dpo/releases/oellm-9b-threeway-dpo-full-v1
CONTAINER=/scratch/project_465002530/users/bmoell/containers/laif-rocm-6.4.4-pytorch-2.9.1-te-2.4.0-fa-2.8.0-triton-3.2.0.sif
OVERLAY=/scratch/project_465002530/users/bmoell/pylibs-overlay
export HF_HOME=/flash/project_465002530/users/bmoell/helpsteer3-dpo/artifacts/cache/huggingface
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$OVERLAY:${PYTHONPATH:-}"
export OMP_NUM_THREADS=7

: "${TAG:?Set an immutable result tag}"
: "${LENGTHS:=512,2048,8192}"
: "${DEPTHS:=50}"
: "${SEEDS:=1}"
mkdir -p "$ROOT/runs" "$ROOT/logs"
test -s "$SFT/model.safetensors"
test -s "$DPO/model.safetensors.index.json"
test -s "$DPO/tokenizer.model"
echo "job=$SLURM_JOB_ID tag=$TAG lengths=$LENGTHS depths=$DEPTHS seeds=$SEEDS"
echo "sft_revision=85bf18fb4f0bee6ac6270f06b1d1c6b3be200f31 dpo_revision=df4ca88030163946040f34b7f238503707906a97"
srun --label singularity exec -B /pfs,/scratch,/flash,/project,/projappl,/appl,/opt/cray,/var/spool/slurmd "$CONTAINER" \
  python3 "$ROOT/scripts/compare_dpo_needle.py" \
    --tokenizer "$DPO" \
    --model "sft=$SFT" --model "dpo=$DPO" \
    --lengths "$LENGTHS" --depths "$DEPTHS" --seeds "$SEEDS" \
    --prompt-format chat --attn-implementation flash_attention_2 \
    --out "$ROOT/runs/$TAG.json"
