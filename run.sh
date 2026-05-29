#!/bin/bash
#
# Usage: ./run.sh <mode> <model_size> [steps] [nodes] [--mbs=N] [--offload|--no-offload]
#
# Modes:     throughput  (50 steps, with W&B)
#            train       (N steps, with W&B and Tensorboard)
#
# Sizes:     125m, 350m, 760m, 1.5b, 3b, 8b
#
# Steps:     required for train mode (e.g., 1000, 5000, 15000)
# Nodes:     unused directly; training uses $SLURM_NNODES from the environment
#
# Flags:     --mbs=N        override the per-model default micro-batch size
#            --offload      enable CPU activation offloading (default)
#            --no-offload   disable CPU activation offloading
#
# Run directly on a debug node (via salloc), or called by launch.sh via srun.

set -euo pipefail

source "$(dirname "$0")/config.sh"

MODE=${1:?Usage: ./run.sh <mode> <model_size> [steps] [nodes]}
MODEL_SIZE=${2:?Usage: ./run.sh <mode> <model_size> [steps] [nodes]}

################ Optional flag parsing ################
MBS_OVERRIDE=""
OFFLOAD_ENABLED=true
for arg in "$@"; do
    case $arg in
        --mbs=*)      MBS_OVERRIDE="${arg#--mbs=}" ;;
        --no-offload) OFFLOAD_ENABLED=false ;;
        --offload)    OFFLOAD_ENABLED=true ;;
    esac
done

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${3:-50}
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        WANDB=true
        ;;
    train)
        TRAINING_STEPS=${3:?Usage: ./run.sh train <model_size> <steps> [nodes]}
        EVAL_INTERVAL=1000
        EVAL_ITERS=10
        LR_WARMUP_ITERS=200
        WANDB=true
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: throughput, train"
        exit 1
        ;;
esac

################ Model config ################
case $MODEL_SIZE in
    125m)
        NUM_LAYERS=12;  HIDDEN=768;  FFN=2048;  HEADS=12; KV_HEADS=4
        MBS=32
        ;;
    350m)
        NUM_LAYERS=24; HIDDEN=1024; FFN=2816;  HEADS=16; KV_HEADS=4
        MBS=16
        ;;
    760m)
        NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4
        MBS=8
        ;;
    1.5b)
        NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4
        MBS=8
        ;;
    3b)
        NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8
        MBS=8
        ;;
    8b)
        NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8
        MBS=4
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m, 350m, 760m, 1.5b, 3b, 8b"
        exit 1
        ;;
esac

if [ -n "$MBS_OVERRIDE" ]; then MBS=$MBS_OVERRIDE; fi

if [ "$OFFLOAD_ENABLED" = true ]; then
    OFFLOAD_LABEL="offload"
    CPU_OFFLOAD_ARGS=(--cpu-offloading-num-layers 23 --use-torch-optimizer-for-cpu-offload --overlap-cpu-optimizer-d2h-h2d)
else
    OFFLOAD_LABEL="nooffload"
    CPU_OFFLOAD_ARGS=()
fi

# GBS=240
GBS=256
SEQ_LEN=4096

################ Paths ################
MEGATRON_LM_DIR=$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/cache

################ Logging paths ################
PROJECT_NAME=gipfelsturm
EXP_NAME="${MODE}-${MODEL_SIZE}-mbs${MBS}-${OFFLOAD_LABEL}-${SLURM_NNODES:-1}n"
LOG_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/$PROJECT_NAME/$EXP_NAME
TENSORBOARD_DIR=$LOG_DIR/tensorboard

echo "START TIME: $(date)"

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR
# flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export NVTE_CPU_OFFLOAD_V1=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-288} / ${SLURM_GPUS_PER_NODE:-4} ))
MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostname "${SLURM_NODELIST:-}" 2>/dev/null | head -1 || hostname)}
MASTER_PORT=25678

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

NETWORK_SIZE_ARGS=(
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN
    --ffn-hidden-size $FFN
    --num-attention-heads $HEADS
    --group-query-attention
    --num-query-groups $KV_HEADS
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length $SEQ_LEN
)

TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-iters $TRAINING_STEPS
    --log-interval 1
    --eval-interval $EVAL_INTERVAL
    --eval-iters $EVAL_ITERS
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 50
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --lr-decay-style constant
    --lr-warmup-iters $LR_WARMUP_ITERS
)

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
)
if [ "$MODE" = "train" ]; then
    LOGGING_ARGS+=(
        --tensorboard-dir $TENSORBOARD_DIR
        --log-timers-to-tensorboard
        --log-memory-to-tensorboard
    )
fi

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 1
)

TORCHRUN_ARGS=(
    --nproc-per-node ${SLURM_GPUS_PER_NODE:-4}
    --nnodes ${SLURM_NNODES:-1}
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${CPU_OFFLOAD_ARGS[@]+"${CPU_OFFLOAD_ARGS[@]}"} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

################ W&B ################
if [ "$WANDB" = true ]; then
    if [ -n "${WANDB_API_KEY:-}" ]; then
        echo "[$(date)] WANDB enabled."
        TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-${SLURM_JOB_ID:-local}"
    else
        export WANDB_MODE=disabled
        echo "[$(date)] WANDB disabled."
    fi
else
    export WANDB_MODE=disabled
fi

echo "CMD: $TRAINING_CMD"
numactl --membind=0-3 $TRAINING_CMD

echo "END TIME: $(date)"
