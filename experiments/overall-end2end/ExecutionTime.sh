#!/bin/bash
#SBATCH --job-name="Execution Time for Long Exposure"
#SBATCH --partition=a100_80
#SBATCH --exclusive
#SBATCH -A chess

#SBATCH --nodes=1
#SBATCH --gres=gpu:7
#SBATCH --ntasks=7  # Number of parallel tasks (executables) you want to run
#SBATCH --cpus-per-task=1  # Number of CPUs per task

#SBATCH --time=08:00:00
#SBATCH --output=ExecutionTime_A100_80GB.out
#SBATCH --error=ExecutionTime_A100_80GB.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alvin.hoang@pnnl.gov

#Dependencies
module load cmake/3.29.0 gcc/11.2.0 python/miniconda24.4.0 gdb/13.1 git/2.42.0
source /share/apps/python/miniconda24.4.0/etc/profile.d/conda.sh
conda activate LLMOptimize

#Command Line
echo -e "Gathering average execution times for systems. \n"

#Batch size 4
#Train 1 optimization everyday
#Training Data dataset/train_gpt.jsonl

# A100 GPT2 137M 512
echo -e "\nA100 GPT2 137M 512. \n"

srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/torch_full.py --model_name openai-community/gpt2 --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/torch_lora.py --model_name openai-community/gpt2 --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/torch_adapter.py --model_name openai-community/gpt2 --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/torch_bitfit.py --model_name openai-community/gpt2 --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/exposer_lora.py --model_name openai-community/gpt2 --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/exposer_adapter.py --model_name openai-community/gpt2 --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/exposer_bitfit.py --model_name openai-community/gpt2 --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl &
wait

# A100 GPT2 137M 1024
echo -e "\nA100 GPT2 137M 1024. \n"

srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/torch_full.py --model_name openai-community/gpt2 --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/torch_lora.py --model_name openai-community/gpt2 --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/torch_adapter.py --model_name openai-community/gpt2 --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/torch_bitfit.py --model_name openai-community/gpt2 --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/exposer_lora.py --model_name openai-community/gpt2 --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/exposer_adapter.py --model_name openai-community/gpt2 --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/exposer_bitfit.py --model_name openai-community/gpt2 --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl &
wait

# A100 GPT2-XL 1.61B 512
echo -e "\nA100 GPT2-XL 1.61B 512. \n"

srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/torch_full.py --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/torch_lora.py --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/torch_adapter.py --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/torch_bitfit.py --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/exposer_lora.py --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/exposer_adapter.py --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/exposer_bitfit.py --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 4 --data dataset/train_gpt.jsonl &
wait

# A100 GPT2-XL 1.61B 1024
echo -e "\nA100 GPT2-XL 1.61B 1024. \n"

srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/torch_full.py --model_name openai-community/gpt2-xl --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/torch_lora.py --model_name openai-community/gpt2-xl --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/torch_adapter.py --model_name openai-community/gpt2-xl --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/torch_bitfit.py --model_name openai-community/gpt2-xl --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/exposer_lora.py --model_name openai-community/gpt2-xl --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/exposer_adapter.py --model_name openai-community/gpt2-xl --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-end2end/exposer_bitfit.py --model_name openai-community/gpt2-xl --seq_len 1024 --batch_size 4 --data dataset/train_gpt.jsonl &
wait

echo -e "\nGathering average execution times complete.\n"
conda deactivate

