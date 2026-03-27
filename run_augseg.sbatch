#!/bin/bash
#SBATCH --job-name=augseg_ptuned
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=training_logs/augseg_p_%j.out
#SBATCH --error=training_logs/augseg_p_%j.err

module purge
module load python/pytorch2.6py3.12

export PATH="$HOME/.local/bin:$PATH"

cd $HOME/Projects/AugSeg

echo "Running on node: $(hostname)"

python - <<'EOF2'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF2

bash scripts/zsing_run_voc_shape.sh


