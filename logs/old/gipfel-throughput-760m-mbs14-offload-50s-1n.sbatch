#!/bin/bash
#SBATCH --account=lsaie-ss26
#SBATCH --time=00:02:00
#SBATCH --job-name=gipfel-throughput-760m-mbs14-offload-50s-1n
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue

echo "START TIME: $(date)"
MASTER_ADDR=$(hostname)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 \
    /users/course_00222/lsaie-ss26-gipfelsturm/run.sh throughput 760m 50 1 --mbs=14 --offload
echo "END TIME: $(date)"
