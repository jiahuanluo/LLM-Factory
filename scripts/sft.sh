#!/bin/bash

# ==============================================================================
# 自动调参微调脚本 (SFT Loop Script)
# ==============================================================================

for bs in 1024 2048 512
do
    for learning_rate in 1e-5 5e-6 5e-5 2e-5
    do
        # 1. 计算梯度累积步数 (Gradient Accumulation Steps)
        accumulation=$((bs / 8 / per_device_train_batch_size))
        
        # 【新增防御逻辑】防止整数除法导致 accumulation 变为 0 进而引发 DeepSpeed 报错
        if [ $accumulation -lt 1 ]; then
            accumulation=1
        fi
        
        # 2. 创建本次实验的输出保存路径
        SAVE="${model_path}/${DATE}/sft_qyz5_${max_seq_length}_${bs}_${learning_rate}_new"
        mkdir -p "${SAVE}"
        
        # 3. 硬件环境自适应判断 (NPU 华为昇腾 vs GPU 显卡)
        if command -v npu-smi &> /dev/null; then
            cp ./modeling_hw.py "${model_path}/modeling.py"
            sed -i 's/"unpad_inputs": true/"unpad_inputs": false/' "${model_path}/config.json"
            sed -i 's/"use_memory_efficient_attention": true/"use_memory_efficient_attention": false/' "${model_path}/config.json"
        else
            cp ./modeling_GQA.py "${model_path}/modeling.py"
            sed -i 's/"unpad_inputs": false/"unpad_inputs": true/' "${model_path}/config.json"
            sed -i 's/"use_memory_efficient_attention": false/"use_memory_efficient_attention": true/' "${model_path}/config.json"
        fi
        
        # 4. 启动分布式 DeepSpeed 训练任务
        deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port ${MASTER_PORT} run_classification.py \
            --model_name_or_path "${model_path}" \
            --trust_remote_code \
            --train_file /workspace/data/qyz_v5/sft_data/qyz5_sft_train_person_samples_v72.csv \
            --validation_file /workspace/data/qyz_v5/sft_data/qyz5_sft_test_person_samples_v72_qijin_Q_marm_new.csv \
            --test_file /workspace/data/qyz_v5/sft_data/qyz5_sft_test_person_samples_v72_qijin_ta_m6_new.csv \
            --text_column_names pbcg2_text \
            --label_column_name label \
            --shuffle_train_dataset \
            --per_device_train_batch_size ${per_device_train_batch_size} \
            --per_device_eval_batch_size 128 \
            --gradient_accumulation_steps ${accumulation} \
            --learning_rate ${learning_rate} \
            --do_train \
            --do_eval \
            --do_regression False \
            --seed ${seed} \
            --max_train_samples 100000000 \
            --max_eval_samples 100000000 \
            --output_dir "${SAVE}" \
            --overwrite_output_dir \
            --logging_steps 10 \
            --max_seq_length ${max_seq_length} \
            --fp16 \
            --num_train_epochs 3 \
            --preprocessing_num_workers 40 \
            --save_strategy steps \
            --eval_strategy steps \
            --save_steps $((100 * 2048 / bs)) \
            --eval_steps $((100 * 2048 / bs)) \
            --eval_delay $((100 * 2048 / bs)) \
            --save_only_model \
            --save_total_limit 3 \
            --metric_name auc \
            --metric_for_best_model eval_auc \
            --ddp_timeout 18000000 \
            --report_to tensorboard \
            --pad_to_max_length \
            --neftune_noise_alpha 0.2 \
            --ddp_find_unused_parameters False \
            --save_safetensors False \
            2>&1 | tee "${SAVE}/log.txt"
            
    done
done