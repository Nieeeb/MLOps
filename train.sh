#!/bin/bash

#SBATCH --job-name=train_CIFAR10_Model
#SBATCH --time=11:57:00
#SBATCH --signal=B:SIGTERM@60
#SBATCH --cpus-per-task=15
#SBATCH --mem=128G

#####################################################################################

# tweak this to fit your needs
max_restarts=24
num_gpus=2
args_file="configs/params.yaml"
#SBATCH --nodelist=ailab-l4-05

# tweak settings to match set parameters
#SBATCH --ntasks=$num_gpus
#SBATCH --gres=gpu:$num_gpus

# Fetch the current restarts value from the job context
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*' | cut -d= -f2)

# If no restarts found, it's the first run, so set restarts to 0
iteration=${restarts:-0}

# Dynamically set output and error filenames using job ID and iteration
outfile="logs/${SLURM_JOB_ID}_${iteration}.out"
errfile="logs/${SLURM_JOB_ID}_${iteration}.err"

# Print the filenames for debugging
echo "Output file: ${outfile}"
echo "Error file: ${errfile}"

##  Define a term-handler function to be executed           ##
##  when the job gets the SIGTERM (before timeout)          ##

term_handler()
{
    echo "Executing term handler at $(date)"
    if [[ $restarts -lt $max_restarts ]]; then
        # Requeue the job, allowing it to restart with incremented iteration
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    else
        echo "Maximum restarts reached, exiting."
        exit 1
    fi
}

# Trap SIGTERM to execute the term_handler when the job gets terminated
trap 'term_handler' SIGTERM

#######################################################################################

# Use srun to dynamically specify the output and error files
srun --output="${outfile}" --error="${errfile}" singularity exec --nv /ceph/project/DAKI4-thermal-2025/MLops/mlops.sif python train_ddp.py --args_file=$args_file --world_size=$num_gpus