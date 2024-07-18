#!/bin/bash
#SBATCH --job-name="Llama3_GPU_Profiling_Experiment"
#SBATCH --partition=dlv
#SBATCH --exclusive
#SBATCH -n 2
#SBATCH --gres=gpu:2  
#SBATCH -A chess
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=ProfileTimeGPU.out
#SBATCH --error=ProfileTimeGPU.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alvin.hoang@pnnl.gov

#Dependencies
module load cmake/3.29.0 gcc/11.2.0 python/3.11.5 cuda/12.3 gdb/13.1

#Command Line
echo -e "Profiling Llama 3 Q8 & Q4 on GPU cluster \n"

# srun --exclusive -n 1 --gres=gpu:1 ./llama-bench -m ./Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf --n-prompt 512 -r 10 &
# srun --exclusive -n 1 --gres=gpu:1 ./llama-bench -m ./Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --n-prompt 512 -r 10 &

srun --exclusive -n 1 --gres=gpu:1 ./LlamaBuildDebug/bin/llama-cli -m ./models/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf 2>errorsFolder/Q8/error 1>outputsFolder/Q8/output -p "What is there to do in Seattle, Washington?" -n 300 -e -ngl 33 -t 16
srun --exclusive -n 1 --gres=gpu:1 gprof ./LlamaBuildDebug/bin/llama-cli 2>errorsFolder/Q8/errorGprof 1>outputsFolder/Q8/outputGprof gmon.out
srun --exclusive -n 1 --gres=gpu:1 ./LlamaBuildDebug/bin/llama-cli -m ./models/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf 2>errorsFolder/Q4/error 1>outputsFolder/Q4/output -p "What is there to do in Seattle, Washington?" -n 300 -e -ngl 33 -t 16
srun --exclusive -n 1 --gres=gpu:1 gprof ./LlamaBuildDebug/bin/llama-cli 2>errorsFolder/Q4/errorGprof 1>outputsFolder/Q4/outputGprof gmon.out

srun --exclusive -n 1 --gres=gpu:1 nvprof ./LlamaBuildDebug/bin/llama-cli -m ./models/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q8_0.gguf 2>errorsFolder/Q8/errorNvprof 1>outputsFolder/Q8/outputNvprof -p "What is there to do in Seattle, Washington?" -n 300 -e -ngl 33 -t 16 &
srun --exclusive -n 1 --gres=gpu:1 nvprof ./LlamaBuildDebug/bin/llama-cli -m ./models/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf 2>errorsFolder/Q4/errorNvprof 1>outputsFolder/Q4/outputNvprof -p "What is there to do in Seattle, Washington?" -n 300 -e -ngl 33 -t 16 &

wait
echo "Profiling of Llama 3 Q8 & Q4 on GPU complete."

