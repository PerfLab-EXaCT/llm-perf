#!/bin/bash
#SBATCH --job-name="Memory Usage for LLM Optimizations"
#SBATCH --partition=a100
#SBATCH --exclusive
#SBATCH -A chess

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1  # Number of parallel tasks (executables) you want to run
#SBATCH --cpus-per-task=1  # Number of CPUs per task

#SBATCH --time=08:00:00
#SBATCH --output=MemoryUsage_A100GB.out
#SBATCH --error=MemoryUsage_A100GB.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alvin.hoang@pnnl.gov

#Dependencies
module load cmake/3.29.0 gcc/11.2.0 python/miniconda24.4.0 gdb/13.1 git/2.42.0
source /share/apps/python/miniconda24.4.0/etc/profile.d/conda.sh
conda activate LLMOptimize

#Command Line
echo -e "Gathering memory usage for systems. \n"

# LoRA
echo -e "LoRA training \n"
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-memory/torch_lora.py --model_name openai-community/gpt2 --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl 
echo -e "\n NEW MEASUREMENT \n"
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-memory/torch_lora.py --model_name openai-community/gpt2 --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl 
echo -e "\n NEW MEASUREMENT \n"
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-memory/torch_lora.py --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl 
echo -e "\n NEW MEASUREMENT \n"
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-memory/torch_lora.py --model_name openai-community/gpt2-xl --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl 

# Full Parameter
echo -e "Full Parameter training \n"
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-memory/torch_full.py --model_name openai-community/gpt2 --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl 
echo -e "\n NEW MEASUREMENT \n"
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-memory/torch_full.py --model_name openai-community/gpt2 --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl 
echo -e "\n NEW MEASUREMENT \n"
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-memory/torch_full.py --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl 
echo -e "\n NEW MEASUREMENT \n"
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-memory/torch_full.py --model_name openai-community/gpt2-xl --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl 

# Adapter
echo -e "Adapter training \n"
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-memory/torch_adapter.py --model_name openai-community/gpt2 --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl 
echo -e "\n NEW MEASUREMENT \n"
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-memory/torch_adapter.py --model_name openai-community/gpt2 --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl 
echo -e "\n NEW MEASUREMENT \n"
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-memory/torch_adapter.py --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl 
echo -e "\n NEW MEASUREMENT \n"
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-memory/torch_adapter.py --model_name openai-community/gpt2-xl --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl 

# BitFit
echo -e "Bitfit training \n"
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-memory/torch_bitfit.py --model_name openai-community/gpt2 --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl 
echo -e "\n NEW MEASUREMENT \n"
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-memory/torch_bitfit.py --model_name openai-community/gpt2 --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl 
echo -e "\n NEW MEASUREMENT \n"
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-memory/torch_bitfit.py --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl 
echo -e "\n NEW MEASUREMENT \n"
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-memory/torch_bitfit.py --model_name openai-community/gpt2-xl --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl 

echo -e "\nGathering memory usage complete.\n"
conda deactivate

