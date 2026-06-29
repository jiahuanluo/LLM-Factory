#!/bin/bash
# =============================================================================
# 超参数网格搜索脚本（参考 scripts/sft.sh 的 bs → accumulation 计算方式）
#
# 核心 —— batch size 三要素的关系:
#   effective_batch_size = per_device_train_batch_size × gradient_accumulation_steps × world_size
#   扫描维度是 effective_batch_size（下方的 BS），
#   accumulation = BS / NUM_GPUS / PER_DEVICE_TRAIN_BATCH_SIZE 自动反推。
#
# 用法: bash scripts/grid_search.sh
# =============================================================================

CONFIG="configs/sft_cls_auc.yaml"
BASE_DIR="output/grid_search"

# ---- 固定项 ----
NUM_GPUS=8                                  # world_size，多卡时和 torchrun --nproc_per_node 一致
PER_DEVICE_TRAIN_BATCH_SIZE=4               # 每卡每步样本数（按显存定）

# ---- 扫描维度 ----
BS=(1024 2048 512)                          # effective batch size（要扫的目标 batch）
LEARNING_RATES=(1e-5 5e-6 5e-5 2e-5)

TOTAL=$((${#BS[@]} * ${#LEARNING_RATES[@]}))
CURRENT=0

for bs in "${BS[@]}"; do
  # accumulation = bs / NUM_GPUS / PER_DEVICE_TRAIN_BATCH_SIZE
  accumulation=$((bs / NUM_GPUS / PER_DEVICE_TRAIN_BATCH_SIZE))
  # 防御: 整数除法变 0 会让 DeepSpeed / Trainer 报错，兜底为 1
  if [ "$accumulation" -lt 1 ]; then
    accumulation=1
  fi
  real_bs=$((PER_DEVICE_TRAIN_BATCH_SIZE * accumulation * NUM_GPUS))
  echo "[batch] target bs=$bs -> per_device=$PER_DEVICE_TRAIN_BATCH_SIZE accumulation=$accumulation world_size=$NUM_GPUS real_effective=$real_bs"

  for lr in "${LEARNING_RATES[@]}"; do
    CURRENT=$((CURRENT + 1))
    OUTPUT_DIR="${BASE_DIR}/bs${bs}_lr${lr}"
    echo "[$CURRENT/$TOTAL] bs=$bs lr=$lr accumulation=$accumulation -> $OUTPUT_DIR"

    mkdir -p "$OUTPUT_DIR"
    torchrun --nproc_per_node "$NUM_GPUS" run_classification.py "$CONFIG" \
      --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
      --gradient_accumulation_steps "$accumulation" \
      --learning_rate "$lr" \
      --output_dir "$OUTPUT_DIR" \
      2>&1 | tee "$OUTPUT_DIR/run.log"

    echo "[$CURRENT/$TOTAL] Done"
    echo ""
  done
done

echo "All $TOTAL experiments completed!"
echo "TensorBoard: tensorboard --logdir $BASE_DIR"
