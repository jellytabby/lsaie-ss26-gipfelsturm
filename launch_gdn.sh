#!/bin/bash
#
# Usage: ./launch_gdn.sh <mode> <model_size> [steps] [nodes] [pattern]
#
# Modes:    throughput  (50 steps default)
#           train       (N steps, with Tensorboard + W&B)
#
# Sizes:    125m, 350m, 760m, 1.5b, 3b, 8b
#           (Same dims as launch.sh so loss/throughput is directly comparable.)
#
# Pattern:  Optional GDN/SDPA layer pattern (default: pure)
#           - pure  : all layers GDN
#           - qwen  : Qwen3-Next style, 1 SDPA every 4 layers (75% GDN)
#           - N     : integer, 1 SDPA every N layers
#           - list  : explicit "[1,0,1,1,1,0,...]" (length == NUM_LAYERS)
#
# Examples: ./launch_gdn.sh throughput 760m
#           ./launch_gdn.sh throughput 1.5b 50 4 qwen
#           ./launch_gdn.sh train 760m 5000 4 pure
#
# Bug-history note (why this script looks the way it does):
#   Previous version exposed `([1]*N)` directly to the shell, which crashed
#   inside srun's `bash -c` with "syntax error near unexpected token '('".
#   We now materialize patterns as `[1,1,1,...]` (no parens, no `*`) and
#   pass the whole argv list as a serialized bash array, so the inner shell
#   never re-parses any value.

set -euo pipefail

source "$(dirname "$0")/config.sh"

MODE=${1:?Usage: ./launch_gdn.sh <mode> <model_size> [steps] [nodes] [pattern]}
MODEL_SIZE=${2:?Usage: ./launch_gdn.sh <mode> <model_size> [steps] [nodes] [pattern]}

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${3:-50}
        NODES=${4:-4}
        TIME=00:18:00
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA=""
        WANDB=true
        ;;
    train)
        TRAINING_STEPS=${3:?Usage: ./launch_gdn.sh train <model_size> <steps> [nodes] [pattern]}
        NODES=${4:-4}
        TIME=02:30:00
        EVAL_INTERVAL=1000
        EVAL_ITERS=10
        LR_WARMUP_ITERS=200
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard"
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
        MBS=2
        ;;
    350m)
        NUM_LAYERS=24; HIDDEN=1024; FFN=2816;  HEADS=16; KV_HEADS=4
        MBS=1
        ;;
    760m)
        NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4
        MBS=1
        ;;
    1.5b)
        NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4
        MBS=1
        ;;
    3b)
        NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8
        MBS=1
        ;;
    8b)
        NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8
        MBS=1
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m, 350m, 760m, 1.5b, 3b, 8b"
        exit 1
        ;;
esac

GBS=256
SEQ_LEN=16384

################ Pattern resolution ################
PATTERN_RAW=${5:-pure}
case $PATTERN_RAW in
    pure)
        # Materialize "[1,1,...,1]" with NUM_LAYERS ones. No parens, no `*`.
        LA_FREQ="[$(printf '1,%.0s' $(seq 1 $NUM_LAYERS) | sed 's/,$//')]"
        PATTERN_TAG="pure"
        ;;
    qwen)
        LA_FREQ="4"
        PATTERN_TAG="qwen"
        ;;
    *[!0-9]*)
        LA_FREQ="$PATTERN_RAW"
        PATTERN_TAG="custom"
        ;;
    *)
        LA_FREQ="$PATTERN_RAW"
        PATTERN_TAG="every${PATTERN_RAW}"
        ;;
esac

JOB_NAME="gipfel-gdn-${MODE}-${MODEL_SIZE}-${PATTERN_TAG}-${TRAINING_STEPS}s-${NODES}n"

echo "[launch_gdn] pattern=$PATTERN_TAG, LA_FREQ='$LA_FREQ'"

################ Generate sbatch script ################
mkdir -p logs
SCRIPT="logs/${JOB_NAME}.sbatch"

# --- HEADER ---
cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

# --- SBATCH directives (expand at generation time) ---
cat >> "$SCRIPT" << SBATCH_DIRECTIVES
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
SBATCH_DIRECTIVES

# --- Static body head ---
cat >> "$SCRIPT" << 'BODY_HEAD'

echo "START TIME: $(date)"

################ Configs ################
BODY_HEAD

# --- Path config (expand at generation time) ---
cat >> "$SCRIPT" << BODY_WORKDIR
WORKDIR=${WORKDIR}
MEGATRON_LM_DIR=\$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/cache
BODY_WORKDIR

# --- Training scalars + GDN linear-attention value
# LA_FREQ is hard-quoted with single quotes so the inner shell sees the bare
# string "[1,1,...]" or "4" and never tries to glob or expand it.
cat >> "$SCRIPT" << CONFIGS

MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}

# Single-quoted on the next line on purpose (no shell parsing of the value).
LA_FREQ='${LA_FREQ}'

PROJECT_NAME=gipfelsturm
EXP_NAME=gdn-${MODE}-${MODEL_SIZE}-${PATTERN_TAG}-\${SLURM_NNODES}n
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CONFIGS

# --- Setup (expanded at runtime) ---
cat >> "$SCRIPT" << 'SETUP'

#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR
ulimit -c 0

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

SETUP

# --- Model args (expand NUM_LAYERS etc. at generation time) ---
cat >> "$SCRIPT" << MODEL
NETWORK_SIZE_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN}
    --num-attention-heads ${HEADS}
    --group-query-attention
    --num-query-groups ${KV_HEADS}
    --max-position-embeddings \$SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length \$SEQ_LEN
)

# GDN args. Note that --linear-attention-freq is passed via \$LA_FREQ
# which holds an unescaped string like "[1,1,...,1]" or "4". As an
# *array element* it stays a single argv token (the quotes around it
# don't end up in argv -- they only protect against shell word-splitting).
GDN_ARGS=(
    --experimental-attention-variant gated_delta_net
    --linear-attention-freq "\$LA_FREQ"
)
MODEL

# --- Training/regularization/LR args ---
cat >> "$SCRIPT" << TRAINING

TRAINING_ARGS=(
    --micro-batch-size \$MBS
    --global-batch-size \$GBS
    --train-iters \$TRAINING_STEPS
    --log-interval 1
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
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
    --lr-warmup-iters ${LR_WARMUP_ITERS}
)
TRAINING

# --- Static args (no expansion) ---
cat >> "$SCRIPT" << 'STATIC_ARGS'

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
STATIC_ARGS

# --- LOGGING_EXTRA + close ---
cat >> "$SCRIPT" << LOGGING_EXTRA
${LOGGING_EXTRA}
)
LOGGING_EXTRA

# --- Tokenizer, data, torchrun, CMD_ARGS assembly ---
cat >> "$SCRIPT" << 'TOKENIZER'

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 0
    --no-create-attention-mask-in-dataloader
)

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

# Build a SINGLE array holding the entire argv. Each "${X[@]}" expansion
# preserves the boundary between elements, so $LA_FREQ stays one token.
CMD_ARGS=(
    "${TORCHRUN_ARGS[@]}"
    "$MEGATRON_LM_DIR/pretrain_gpt.py"
    "${TRANSFORMER_ENGINE_ARGS[@]}"
    "${NETWORK_SIZE_ARGS[@]}"
    "${GDN_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${REGULARIZATION_ARGS[@]}"
    "${LEARNING_RATE_ARGS[@]}"
    "${INITIALIZATION_ARGS[@]}"
    "${MIXED_PRECISION_ARGS[@]}"
    "${DISTRIBUTED_ARGS[@]}"
    "${LOGGING_ARGS[@]}"
    "${TOKENIZER_ARGS[@]}"
    "${DATA_ARGS[@]}"
)

if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    CMD_ARGS+=(
        --wandb-save-dir "$LOG_DIR"
        --wandb-project "$PROJECT_NAME"
        --wandb-exp-name "$EXP_NAME-$SLURM_JOB_ID"
    )
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi

echo "CMD: torchrun ${CMD_ARGS[*]}"

# Serialize the array. The child shell will `eval` this declare-p output to
# rehydrate the array byte-for-byte -- no word-splitting on `[`, `]`, `,`.
CMD_ARGS_SERIALIZED=$(declare -p CMD_ARGS)
export CMD_ARGS_SERIALIZED

srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 \
     --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c '
    if [ "${SLURM_LOCALID:-0}" = "0" ]; then
        pip install flash-linear-attention causal-conv1d tilelang --quiet
        touch /tmp/install_done
    else
        while [ ! -f /tmp/install_done ]; do sleep 1; done
    fi
    export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache/rank_${LOCAL_RANK}
    eval "$CMD_ARGS_SERIALIZED"
    numactl --membind=0-3 torchrun "${CMD_ARGS[@]}"
'
echo "END TIME: $(date)"
TOKENIZER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
sbatch "$SCRIPT"