#!/bin/bash
#SBATCH --job-name="Testing optimal amount of numWorkers"
#SBATCH --partition=a100
#SBATCH --exclusive
#SBATCH -A chess
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4  # Number of parallel tasks (executables) you want to run
#SBATCH --cpus-per-task=1  # Number of CPUs per task
#SBATCH --time=08:00:00
#SBATCH --output=OptimalNumWorker_A100.out
#SBATCH --error=OptimalNumWorker_A100.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alvin.hoang@pnnl.gov

#Dependencies
module load cmake/3.29.0 gcc/11.2.0 python/miniconda24.4.0 gdb/13.1 git/2.42.0
source /share/apps/python/miniconda24.4.0/etc/profile.d/conda.sh
conda activate LLMOptimize

#Command Line
#Batch size 4
#Training Data dataset/train_gpt.jsonl

echo -e "\nGathering average execution times start.\n"

srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-numWorker/numWorkerTest.py --numWorker 0 --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 2 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-numWorker/numWorkerTest.py --numWorker 1 --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 2 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-numWorker/numWorkerTest.py --numWorker 2 --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 2 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-numWorker/numWorkerTest.py --numWorker 4 --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 2 --data dataset/train_gpt.jsonl &
wait

srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-numWorker/numWorkerTest.py --numWorker 8 --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 2 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-numWorker/numWorkerTest.py --numWorker 16 --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 2 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-numWorker/numWorkerTest.py --numWorker 32 --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 2 --data dataset/train_gpt.jsonl &
srun --exclusive -n 1 --gres=gpu:1 python3 experiments/overall-numWorker/numWorkerTest.py --numWorker 64 --model_name openai-community/gpt2-xl --seq_len 512 --batch_size 2 --data dataset/train_gpt.jsonl &
wait

echo -e "\nGathering average execution times complete.\n"
conda deactivate