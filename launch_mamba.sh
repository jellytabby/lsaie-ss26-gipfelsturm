#!/bin/bash
#
# Usage: ./launch_mamba.sh <model_size> <steps> [nodes] [seq_len]
#
# Sizes:    mamba-125m
#
# Steps:    number of training steps (e.g. 5 for param-count check, 50 for throughput)
# Nodes:    optional, default 4
# Seq_len:  optional, default 4096 (also supports 8192, 16384)
#
# Examples: ./launch_mamba.sh mamba-125m 5 1          # param-count check
#           ./launch_mamba.sh mamba-125m 50 4          # throughput benchmark
#           ./launch_mamba.sh mamba-125m 50 4 8192     # throughput at seq_len=8192

set -euo pipefail

source "$(dirname "$0")/config.sh"

MODEL_SIZE=${1:?Usage: ./launch_mamba.sh <model_size> <steps> [nodes] [seq_len]}
TRAINING_STEPS=${2:?Usage: ./launch_mamba.sh <model_size> <steps> [nodes] [seq_len]}
NODES=${3:-4}
SEQ_LEN=${4:-4096}

TIME=00:25:00
EVAL_INTERVAL=$TRAINING_STEPS
EVAL_ITERS=0
LR_WARMUP_ITERS=$(( TRAINING_STEPS < 10 ? 0 : 10 ))

################ Model config ################
case $MODEL_SIZE in
    mamba-125m)
        NUM_LAYERS=12; HIDDEN=768; FFN=2048; HEADS=12
        case $SEQ_LEN in
            4096)  MBS=8 ;; 8192)  MBS=4 ;; 16384) MBS=2 ;; *) MBS=8 ;;
        esac
        ;;
    mamba-350m)
        NUM_LAYERS=24; HIDDEN=1024; FFN=2816;  HEADS=16
        case $SEQ_LEN in
            4096)  MBS=4 ;; 8192)  MBS=2 ;; 16384) MBS=1 ;; *) MBS=4 ;;
        esac
        ;;
    mamba-760m)
        NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16
        case $SEQ_LEN in
            4096)  MBS=4 ;; 8192)  MBS=2 ;; 16384) MBS=1 ;; *) MBS=4 ;;
        esac
        ;;
    mamba-1.5b)
        NUM_LAYERS=48; HIDDEN=1536; FFN=4352;  HEADS=16
        case $SEQ_LEN in
            4096)  MBS=2 ;; 8192)  MBS=1 ;; 16384) MBS=1 ;; *) MBS=2 ;;
        esac
        ;;
    mamba-3b)
        NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24
        case $SEQ_LEN in
            4096)  MBS=2 ;; 8192)  MBS=1 ;; 16384) MBS=1 ;; *) MBS=2 ;;
        esac
        ;;
    mamba-8b)
        NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32
        case $SEQ_LEN in
            4096)  MBS=1 ;; 8192)  MBS=1 ;; 16384) MBS=1 ;; *) MBS=1 ;;
        esac
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: mamba-125m"
        exit 1
        ;;
esac

GBS=256
JOB_NAME="gipfel-mamba-pure-${MODEL_SIZE}-${TRAINING_STEPS}s-${NODES}n-seq${SEQ_LEN}"

################ Generate script ################
mkdir -p logs

SCRIPT="logs/${JOB_NAME}.sbatch"

# --- HEADER ---
cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

# --- SBATCH DIRECTIVES (expand at generation time) ---
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

# --- RUNTIME CONSTANTS (single-quoted: expand at runtime) ---
cat >> "$SCRIPT" << 'BODY_HEAD'

echo "START TIME: $(date)"

################ Configs ################
BODY_HEAD

# --- PATHS (expand at generation time) ---
cat >> "$SCRIPT" << BODY_WORKDIR
WORKDIR=${WORKDIR}
MEGATRON_LM_DIR=\$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/cache
BODY_WORKDIR

# --- TRAINING SCALARS (expand at generation time) ---
cat >> "$SCRIPT" << CONFIGS

MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}

PROJECT_NAME=gipfelsturm
EXP_NAME=mamba-${MODEL_SIZE}-\${SLURM_NNODES}n
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/\$PROJECT_NAME/\$EXP_NAME
CONFIGS

# --- SETUP (single-quoted: paths use runtime \$VAR) ---
cat >> "$SCRIPT" << 'SETUP'

#########################################

mkdir -p logs $LOG_DIR $DATASET_CACHE_DIR
ulimit -c 0

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
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

# --- TRAINING ARGS (expand at generation time where appropriate) ---
cat >> "$SCRIPT" << TRAINING_ARGS

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
TRAINING_ARGS

# --- STATIC ARGS (single-quoted) ---
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
)

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
)

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)
STATIC_ARGS

# --- MAMBA TRAINING CMD
# NUM_LAYERS, HIDDEN, FFN, HEADS expand at generation time (no backslash)
# TORCHRUN_ARGS, SEQ_LEN, MEGATRON_LM_DIR expand at runtime (backslash)
cat >> "$SCRIPT" << MAMBA_CMD

TRAINING_CMD="torchrun \${TORCHRUN_ARGS[@]} \$MEGATRON_LM_DIR/pretrain_mamba.py \
    \${TRANSFORMER_ENGINE_ARGS[@]} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN} \
    --ffn-hidden-size ${FFN} \
    --num-attention-heads ${HEADS} \
    --max-position-embeddings \$SEQ_LEN \
    --position-embedding-type none \
    --normalization RMSNorm \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --seq-length \$SEQ_LEN \
    --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
    --hybrid-attention-ratio 0.0 \
    --hybrid-mlp-ratio 0.0 \
    --no-create-attention-mask-in-dataloader \
    \${TRAINING_ARGS[@]} \
    \${REGULARIZATION_ARGS[@]} \
    \${LEARNING_RATE_ARGS[@]} \
    \${INITIALIZATION_ARGS[@]} \
    \${MIXED_PRECISION_ARGS[@]} \
    \${DISTRIBUTED_ARGS[@]} \
    \${LOGGING_ARGS[@]} \
    \${TOKENIZER_ARGS[@]} \
    \${DATA_ARGS[@]}"
MAMBA_CMD

# --- W&B ---
cat >> "$SCRIPT" << 'WANDB'

if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    TRAINING_CMD="$TRAINING_CMD \
        --wandb-save-dir $LOG_DIR \
        --wandb-project $PROJECT_NAME \
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi
WANDB

# --- FOOTER ---
cat >> "$SCRIPT" << 'FOOTER'

echo "CMD: $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "
    if [ \"\${SLURM_LOCALID:-0}\" = \"0\" ]; then
        pip install mamba-ssm causal-conv1d --no-build-isolation --quiet
        touch /tmp/install_done
    else
        while [ ! -f /tmp/install_done ]; do sleep 1; done
    fi
    export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/.triton_cache/rank_\${LOCAL_RANK}
    numactl --membind=0-3 $TRAINING_CMD
"
echo "END TIME: $(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
sbatch "$SCRIPT"