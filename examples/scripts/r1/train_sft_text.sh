#!/bin/bash

# set -x

# =================== User Configuration ===================
# Please modify these variables according to your environment
# =========================================================

# Base paths - MODIFY THESE
export WORKSPACE_DIR="$(pwd)"                      # Path to project root directory
export DATASET_PATH="./data/am_0.5M.jsonl"  # Path to your dataset
export PRETRAIN_MODEL_PATH="../../Qwen2.5-VL-3B-Instruct"  # Path to pretrained model
export SAVE_PATH="./checkpoints"                   # Absolute path to save checkpoints

# Model configuration
export MODEL_NAME="qwen-2.5-vl-3b-sft-text"              # Name for this training run

# Wandb configuration (optional)
export WANDB_DIR="${WORKSPACE_DIR}"                # Directory for wandb files
export WANDB_API_KEY="dabb4679bdd222ed5d5a2f48741598127d413010"          # Your wandb API key (if online)

# optimize CUDA momory setting
export PYTORCH_CUDA_ALLOC_CONCURRENCY=2  # 并行内存分配
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512,roundup_power2_divisions:4"
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0  # 启用InfiniBand（若集群支持）
export CUDA_LAUNCH_BLOCKING=0  # 异步内核执行

# =================== Preprocess Dataset ===================
#python ./data/sft_preprocess.py

# =================== Script Execution ===================
read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
    --max_len 4096 \
    --dataset ${DATASET_PATH} \
    --input_key question \
    --output_key response \
    --train_batch_size 32 \
    --micro_train_batch_size 4 \
    --max_samples 50000 \
    --pretrain ${PRETRAIN_MODEL_PATH} \
    --save_path ${SAVE_PATH}/${MODEL_NAME} \
    --save_steps -1 \
    --logging_steps 50 \
    --eval_steps 10000 \
    --zero_stage 3 \
    --max_epochs 1 \
    --bf16 \
    --flash_attn \
    --learning_rate 2e-6 \
    --load_checkpoint \
    --gradient_checkpointing \
    --use_wandb ${WANDB_API_KEY} \
    --wandb_run_name ${MODEL_NAME} \
    --wandb_group "r1"
EOF

# --wandb ${WANDB_DIR} \
# --wandb_name "${MODEL_NAME}-$(date +%Y%m%d-%H%M)"

if [[ ${1} != "slurm" ]]; then
    deepspeed  --master_port=29501 --module $training_commands
fi
