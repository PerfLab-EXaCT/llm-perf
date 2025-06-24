#!/bin/bash
#SBATCH --job-name="Get Padding"
#SBATCH --partition=h100
#SBATCH --exclusive
#SBATCH -A compress
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=pad.out
#SBATCH --error=pad.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alvin.hoang@pnnl.gov

# Dependencies
module load gcc/11.2.0 python/miniconda24.4.0 git/2.42.0 cuda/12.1
source /share/apps/python/miniconda24.4.0/etc/profile.d/conda.sh
conda activate BatchTest

squeue -u hoan163

echo -e "Warmup Run"
srun --exclusive python SmartBatch/finetune_single.py --num_epochs 1

echo -e "\n\nPadding with Group By Length"
srun --exclusive python SmartBatch/finetune_single.py --group_by_length --num_epochs 1

echo -e "\n\nPadding with Smart Batching"
srun --exclusive python SmartBatch/finetune_single.py --group_by_length --no_shuffle --num_epochs 1

# echo -e "\n\nPadding with Group By Length (Worst Case)"
# srun --exclusive python SmartBatch/finetune_single.py --group_by_length --worst_case --num_epochs 1

# Deactivate conda environment
conda deactivate