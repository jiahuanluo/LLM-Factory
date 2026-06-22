#!/bin/bash
# =============================================================================
# 超参数网格搜索脚本
# 用法: bash scripts/grid_search.sh
# 多卡并行: CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/grid_search.sh
# =============================================================================

CONFIG="configs/sft_cls_auc.yaml"
BASE_DIR="output/grid_search"

# 定义搜索空间
LEARNING_RATES=(1e-5 2e-5 5e-5)
BATCH_SIZES=(16 32 64)

# 计算总组合数
TOTAL=$((${#LEARNING_RATES[@]} * ${#BATCH_SIZES[@]}))
CURRENT=0

for lr in "${LEARNING_RATES[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    CURRENT=$((CURRENT + 1))
    OUTPUT_DIR="${BASE_DIR}/lr${lr}_bs${bs}"

    echo "[$CURRENT/$TOTAL] lr=$lr, bs=$bs -> $OUTPUT_DIR"

    python run_classification.py "$CONFIG" \
      --learning_rate "$lr" \
      --per_device_train_batch_size "$bs" \
      --output_dir "$OUTPUT_DIR" \
      --report_to tensorboard

    echo "[$CURRENT/$TOTAL] Done"
    echo ""
  done
done

echo "All $TOTAL experiments completed!"
echo "Results saved to $BASE_DIR/"
echo ""
echo "View all results in TensorBoard:"
echo "  tensorboard --logdir $BASE_DIR"
