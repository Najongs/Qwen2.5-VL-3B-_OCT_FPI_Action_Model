#!/bin/bash
# ================================================================
# 🔹 Multi-GPU DDP launcher for VLA Training
# ================================================================

# ✅ GPU 개수 설정
NUM_GPUS=4

# ✅ 실행할 학습 스크립트
TRAIN_SCRIPT="5st_VLA_TRAIN_VL_Lora.py"

# ✅ 로그 디렉토리 설정
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# ================================================================
# 🔹 PyTorch 환경 세팅
# ================================================================
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONWARNINGS="ignore"

# ================================================================
# 🔹 하이퍼파라미터 설정
# ================================================================
LR=5e-5
MIN_LR=1e-8
WARMUP_RATIO=0.05
HOLD_RATIO=0.02
GRAD_ACCUM=8
SCHED_ON="step"

# ================================================================
# 🔹 1. Cache 생성 실행
# ================================================================
# CACHE_LOG_FILE="$LOG_DIR/cache_$(date +%Y%m%d_%H%M%S).log"
# echo "🚀 [STEP 1/2] Launching Caching on $NUM_GPUS GPUs"
# echo "🔧 Log file: $CACHE_LOG_FILE"
# echo "================================================"

# torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
#     $TRAIN_SCRIPT \
#     --mode cache \
#     --lr $LR \
#     --min-lr $MIN_LR \
#     --warmup-ratio $WARMUP_RATIO \
#     --hold-ratio $HOLD_RATIO \
#     --grad-accum-steps $GRAD_ACCUM \
#     --sched-on $SCHED_ON \
#     2>&1 | tee $CACHE_LOG_FILE

# echo "✅ Caching finished at $(date)"
# echo ""
# echo "================================================"
# echo "================================================"
# echo ""

# ================================================================
# 🔹 2. 학습 실행
# ================================================================
TRAIN_LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
echo "🚀 [STEP 2/2] Launching Training on $NUM_GPUS GPUs"
echo "🔧 Log file: $TRAIN_LOG_FILE"
echo "================================================"

# torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
#     $TRAIN_SCRIPT \
#     --mode train \
#     --lr $LR \
#     --min-lr $MIN_LR \
#     --warmup-ratio $WARMUP_RATIO \
#     --hold-ratio $HOLD_RATIO \
#     --grad-accum-steps $GRAD_ACCUM \
#     --sched-on $SCHED_ON \
#     2>&1 | tee $TRAIN_LOG_FILE


torchrun --nproc_per_node=4 5st_VLA_TRAIN_VL_Lora.py \
    --mode train \
    --finetune-vl lora \
    --lr 5e-4 \
    --vl-lr 1e-5 \
    --vision-lr 5e-6 \
    --grad-accum-steps 64 \
    --sched-on step


echo "✅ Training finished at $(date)"