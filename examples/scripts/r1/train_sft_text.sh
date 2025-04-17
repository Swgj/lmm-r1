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
    --micro_train_batch_size 2 \
    --max_samples 500000 \
    --pretrain ${PRETRAIN_MODEL_PATH} \
    --save_path ${SAVE_PATH}/${MODEL_NAME} \
    --save_steps 2500 \
   --logging_steps 10 \
   --eval_steps 1000 \
   --zero_stage 2 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 2e-6 \
   --load_checkpoint \
   --gradient_checkpointing
EOF

# --wandb ${WANDB_DIR} \
# --wandb_name "${MODEL_NAME}-$(date +%Y%m%d-%H%M)"

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
