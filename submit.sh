#!/bin/bash
# Wrapper to submit Slurm jobs with automatic log naming.
#
# Usage: ./submit.sh <script.py> [args...]
# Example:
#   ./submit.sh rnn/train.py --epochs 10
#   ./submit.sh rnn/train.py --epochs 20 --lr 0.0005 --hidden-size 256

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <script.py> [args...]"
    exit 1
fi

SCRIPT="$1"
shift
EXTRA_ARGS="$@"

# Derive a clean job name from the script filename
JOB_NAME="$(basename "$SCRIPT" .py)"

mkdir -p logs

sbatch \
    --parsable \
    --job-name="$JOB_NAME" \
    --output="logs/${JOB_NAME}_%j.log" \
    --error="logs/${JOB_NAME}_%j.err" \
    slurm_gpu.sh "$SCRIPT" $EXTRA_ARGS \
    | {
        read -r SBATCH_OUTPUT
        JOB_ID="${SBATCH_OUTPUT%%;*}"

        echo "Submitted batch job ${JOB_ID}"
        echo "Watch logs with:"
        echo "  tail -f logs/${JOB_NAME}_${JOB_ID}.log logs/${JOB_NAME}_${JOB_ID}.err"
    }
