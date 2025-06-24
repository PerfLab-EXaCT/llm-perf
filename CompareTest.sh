#!/bin/bash
#SBATCH --job-name="Runtime Measurement"
#SBATCH --partition=h100
#SBATCH --exclusive
#SBATCH -A compress
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=test_32.out
#SBATCH --error=test_32.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=alvin.hoang@pnnl.gov

# Dependencies
module load cmake/3.29.0 gcc/11.2.0 python/miniconda24.4.0 gdb/13.1 git/2.42.0 cuda/12.1
source /share/apps/python/miniconda24.4.0/etc/profile.d/conda.sh
conda activate BatchTest

squeue -u hoan163

# Get average runtime of finetuning steps
echo -e "\n Warmup Run \n"
srun --exclusive -n 1 --gres=gpu:1 python /people/hoan163/project/finetune_single.py --num_epochs 1

# No Group by length
echo -e "\n No Group_By_Length Runtime \n"
average_no_group_by_length=0
for i in {1..5}; do
    echo "Run $i of 5 (No Group By Length)"
    runtime=$(srun --exclusive -n 1 --gres=gpu:1 python /people/hoan163/project/finetune_single.py | grep "Runtime:" | cut -d':' -f2 | xargs)
    echo "Runtime for run $i: $runtime"
    average_no_group_by_length=$(echo "$average_no_group_by_length + $runtime" | bc -l)
done
average_no_group_by_length=$(echo "scale=4; $average_no_group_by_length / 5" | bc -l)

# Group by length
echo -e "\n Group_By_Length Runtime \n"
average_group_by_length=0
for i in {1..5}; do
    echo "Run $i of 5 (Group By Length)"
    runtime=$(srun --exclusive -n 1 --gres=gpu:1 python /people/hoan163/project/finetune_single.py --group_by_length | grep "Runtime:" | cut -d':' -f2 | xargs)
    echo "Runtime for run $i: $runtime"
    average_group_by_length=$(echo "$average_group_by_length + $runtime" | bc -l)
done
average_group_by_length=$(echo "scale=4; $average_group_by_length / 5" | bc -l)

# Smart Batch
echo -e "\n Smart Batch Runtime \n"
average_smart_batch=0
for i in {1..5}; do
    echo "Run $i of 5 (Smart Batch)"
    runtime=$(srun --exclusive -n 1 --gres=gpu:1 python /people/hoan163/project/finetune_single.py --group_by_length --no_shuffle | grep "Runtime:" | cut -d':' -f2 | xargs)
    echo "Runtime for run $i: $runtime"
    average_smart_batch=$(echo "$average_smart_batch + $runtime" | bc -l)
done
average_smart_batch=$(echo "scale=4; $average_smart_batch / 5" | bc -l)

# Deactivate conda environment
conda deactivate

echo -e "\n=== FINAL RESULTS ==="
echo -e "No Group_By_Length Average Runtime: $average_no_group_by_length seconds"
echo -e "Group_By_Length Average Runtime: $average_group_by_length seconds"
echo -e "Smart_Batch Average Runtime: $average_smart_batch seconds"