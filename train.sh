#!/bin/bash

model=${1:-"stable-diffusion-v1-5/stable-diffusion-v1-5"}  # which diffusion model to fine tune
batch_size=${2:-16}  # per-GPU batch size (adjust based on your GPU memory)
rank=${3:-8}  # LoRA rank
learning_rate=${4:-1e-4}  # learning rate
seed=${5:-"42"}  # random seed

num_gpus=$(nvidia-smi -L | wc -l)



accelerate launch \
    --num_machines=1 \
    --num_processes="$num_gpus" \
    --mixed_precision=fp16 \
    --gpu_ids=all \
    --main_training_function=main \
    --dynamo_backend="no" \
    train_text_to_image_lora.py \
        --pretrained_model_name_or_path="${model}" \
        --rank=${rank} \
        --train_data_dir="data/train" \
        --caption_column="prompt" \
        --resolution=512 \
        --center_crop \
        --random_flip \
        --num_train_epochs=50 \
        --train_batch_size=$batch_size \
        --gradient_accumulation_steps=$((16 / $batch_size)) \
        --learning_rate="${learning_rate}" \
        --lr_scheduler="cosine" \
        --lr_warmup_steps=50 \
        --seed=${seed} \
        --validation_prompt="a portrait photo of a DESC, aged AGE, realistic, high quality" \
        --num_validation_images=8 \
        --checkpointing_epochs=10 \
        --output_dir="train_output" \
        --report_to="wandb" \
        --mixed_precision="fp16" \
        --use_8bit_adam \
        --enable_xformers_memory_efficient_attention \