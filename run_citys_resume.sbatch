#!/bin/bash
#SBATCH --job-name=augseg_citys_resume
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=training_logs/citys_resume_%j.out
#SBATCH --error=training_logs/citys_resume_%j.err

module purge
module load python/pytorch2.6py3.12

export PATH="$HOME/.local/bin:$PATH"

cd $HOME/Projects/AugSeg

echo "Running on node: $(hostname)"

python - <<'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

#bash scripts/zsing_run_citys.sh
bash scripts/zsing_run_citys_resume.sh

