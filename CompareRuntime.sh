#!/bin/bash 
#SBATCH --job-name="Runtime Measurement"   #? This is the title of the job you want to run
#SBATCH --partition=h100     #? This is the partition/cluster you want to run the job on. In your case, you want a100 or h100. To see all partitions/clusters, run "sinfo" or check Deception website.
#SBATCH --exclusive     #? This means you want to use all the resources on the node for your job only. If you want to share the node with other jobs, remove this line
#SBATCH -A compress   #? This is the project/allocation you want to charge the job to. 

#SBATCH --nodes=1   #? This is the number of nodes you want to use. 
#SBATCH --gres=gpu:1  #? This is the number of GPUs you want to use. If you want to use more than one GPU, change this number.
#SBATCH --time=08:00:00     #? This is the maximum time you want the job to run. If the job runs longer than this time, it will stop the job.
#SBATCH --output=test_32.out    #? This is the output file where you want to save the output of the job.
#SBATCH --error=test_32.err     #? This is the error file where you want to save the errors of the job.
#SBATCH --mail-type=ALL         #? This is the type of email you want to receive if the job starts, ends, or fails. If you don't want to receive any emails, you can remove this line.
#SBATCH --mail-user=alvin.hoang@pnnl.gov    #? This is where you put your email address where you receive emails letting you know your job finished. If you don't want to receive any emails, you can remove this line.

#Dependencies   #? Load all modules and dependencies here. Also activate conda environment if you are using one.
module load cmake/3.29.0 gcc/11.2.0 python/miniconda24.4.0 gdb/13.1 git/2.42.0 cuda/12.1
source /share/apps/python/miniconda24.4.0/etc/profile.d/conda.sh
conda activate BatchTest

squeue -u hoan163

#Get average runtime of finetuning steps 
echo -e "\n Warmup Run \n"
srun --exclusive -n 1 --gres=gpu:1 python /people/hoan163/project/finetune_single.py --num_epochs 1

#No Group by length
echo -e "\n No Group_By_Length Runtime \n"
average_no_group_by_length=0
for i in {1..5}
do
  srun --exclusive -n 1 --gres=gpu:1 python /people/hoan163/project/finetune_single.py
  #average_no_group_by_length=$(echo "$average_no_group_by_length + $runtime" | bc -l)
done
#average_no_group_by_length=$(echo "$average_no_group_by_length / 5" | bc -l)

#Group by length
echo -e "\n Group_By_Length Runtime \n"
average_group_by_length=0
for i in {1..5}
do
  srun --exclusive -n 1 --gres=gpu:1 python /people/hoan163/project/finetune_single.py --group_by_length
  #average_group_by_length=$(echo "$average_group_by_length + $runtime" | bc -l)
done
#average_group_by_length=$(echo "$average_group_by_length / 5" | bc -l)

#Smart Batch
echo -e "\n Smart Batch \n"
average_smart_batch=0
for i in {1..5}
do
  srun --exclusive -n 1 --gres=gpu:1 python /people/hoan163/project/finetune_single.py --group_by_length --no_shuffle
  #average_smart_batch=$(echo "$average_smart_batch + $runtime" | bc -l)
done
#average_smart_batch=$(echo "$average_smart_batch / 5" | bc -l)

# Deactivate conda environment
conda deactivate

# echo -e "No Group_By_Length Average Runtime: $average_no_group_by_length"
# echo -e "Group_By_Length Average Runtime: $average_group_by_length"
# echo -e "Smart_Batch Average Runtime: $average_smart_batch"

