#!/bin/bash
#SBATCH --job-name=rnn_gpu
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h200-141:1

# Usage: sbatch slurm_gpu.sh <script.py> [args...]
# Example: sbatch slurm_gpu.sh rnn/train.py --epochs 10

if [ -z "$1" ]; then
    echo "Error: No script provided."
    echo "Usage: sbatch slurm_gpu.sh <script.py> [args...]"
    exit 1
fi

SCRIPT="$1"
shift
SCRIPT_ARGS="$@"

cd ~/cs3244-group14
source ~/venv/bin/activate

echo "Running: python $SCRIPT $SCRIPT_ARGS"
echo "Device: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no GPU detected')"

python "$SCRIPT" $SCRIPT_ARGS
