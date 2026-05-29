#!/bin/bash
#
# Usage: ./launch.sh <mode> <model_size> [steps] [nodes] [--mbs=N] [--offload|--no-offload]
#
# Modes:     throughput  (50 steps, with W&B)
#            train       (N steps, with W&B and Tensorboard)
#
# Sizes:     125m, 350m, 760m, 1.5b, 3b, 8b
#
# Steps:     required for train mode (e.g., 1000, 5000, 15000)
# Nodes:     optional, default 4 (max 8)
#
# Flags:     --mbs=N        override the per-model default micro-batch size
#            --offload      enable CPU activation offloading (default)
#            --no-offload   disable CPU activation offloading
#
# Examples:  ./launch.sh throughput 760m
#            ./launch.sh throughput 8b 50 1
#            ./launch.sh train 760m 5000
#            ./launch.sh train 1.5b 3000 8
#            ./launch.sh throughput 760m 50 1 --mbs=6 --no-offload
#
# For interactive use on a debug node, run run.sh directly instead.

set -euo pipefail

source "$(dirname "$0")/config.sh"

MODE=${1:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes]}
MODEL_SIZE=${2:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes]}

################ Optional flag parsing (for job name only) ################
MBS_OVERRIDE=""
OFFLOAD_ENABLED=true
for arg in "$@"; do
    case $arg in
        --mbs=*)      MBS_OVERRIDE="${arg#--mbs=}" ;;
        --no-offload) OFFLOAD_ENABLED=false ;;
        --offload)    OFFLOAD_ENABLED=true ;;
    esac
done

################ Mode config (for SBATCH directives) ################
case $MODE in
    throughput)
        TRAINING_STEPS=${3:-50}
        NODES=${4:-4}
        TIME=00:15:00
        ;;
    train)
        TRAINING_STEPS=${3:?Usage: ./launch.sh train <model_size> <steps> [nodes]}
        NODES=${4:-4}
        TIME=02:30:00
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: throughput, train"
        exit 1
        ;;
esac

################ Model config (for job name only) ################
case $MODEL_SIZE in
    125m) MBS=32 ;;
    350m) MBS=16 ;;
    760m) MBS=8  ;;
    1.5b) MBS=8  ;;
    3b)   MBS=8  ;;
    8b)   MBS=4  ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m, 350m, 760m, 1.5b, 3b, 8b"
        exit 1
        ;;
esac

if [ -n "$MBS_OVERRIDE" ]; then MBS=$MBS_OVERRIDE; fi
OFFLOAD_LABEL=$([ "$OFFLOAD_ENABLED" = true ] && echo "offload" || echo "nooffload")
JOB_NAME="gipfel-${MODE}-${MODEL_SIZE}-mbs${MBS}-${OFFLOAD_LABEL}-${TRAINING_STEPS}s-${NODES}n"

################ Generate script ################
mkdir -p logs

SCRIPT="logs/${JOB_NAME}.sbatch"

cat > "$SCRIPT" << SBATCH
#!/bin/bash
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue

echo "START TIME: \$(date)"
MASTER_ADDR=\$(hostname)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task \$SLURM_CPUS_PER_TASK --wait 60 \\
    ${WORKDIR}/run.sh $@
echo "END TIME: \$(date)"
SBATCH

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
sbatch "$SCRIPT"
