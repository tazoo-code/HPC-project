#!/bin/bash

seed=${1:-"42"}  # random seed
model=${2:-"stable-diffusion-v1-5/stable-diffusion-v1-5"}
rank=${3:-"8"}  # for wandb logging

batch_size=8
learning_rate=1e-04

accel_config="./accelerate_config.yaml"
cat > "$accel_config" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: $(nvidia-smi -L | wc -l)
mixed_precision: fp16
gpu_ids: all
main_training_function: main
downcast_bf16: 'no'
EOF

accelerate launch \
    --config_file ./accelerate_config.yaml \
    --num_processes $(nvidia-smi -L | wc -l) \
    train_text_to_image_lora.py \
        --pretrained_model_name_or_path="${model}" \
        --train_data_dir="data/train" \
        --caption_column="prompt" \
        --resolution=512 \
        --center_crop \
        --random_flip \
        --train_batch_size=$batch_size \
        --gradient_accumulation_steps=1 \
        --gradient_checkpointing \
        --max_train_steps=5000 \
        --learning_rate=1e-4 \
        --lr_scheduler="cosine" \
        --lr_warmup_steps=200 \
        --rank=$rank \
        --mixed_precision="fp16" \
        --checkpointing_steps=1000 \
        --checkpoints_total_limit=3 \
        --output_dir="train_output" \
        --seed=42 \
        --report_to="wandb"
        # --pretrained_model_name_or_path="${model}" \
        # --rank="${rank}" \
        # --train_data_dir="data/train" \
        # --resolution=512 \
        # --caption_column="prompt" \
        # --center_crop \
        # --random_flip \
        # --num_train_epochs=40 \
        # --train_batch_size="${batch_size}" \
        # --gradient_accumulation_steps=1 \
        # --learning_rate="${learning_rate}" \
        # --lr_scheduler="cosine" \
        # --lr_warmup_steps=500 \
        # --seed="${seed}" \
        # --dataloader_num_workers=4 \
        # --validation_prompt="a portrait photo of a DESC, aged AGE, realistic, high quality" \
        # --num_validation_images=8 \
        # --report_to="wandb" \
        # --checkpointing_steps=2000 \
        # --checkpoints_total_limit=3 \
        # --output_dir=train_outputs \
        # --mixed_precision="fp16" \
        # --max_train_samples=1000 \
        # --use_8bit_adam \
        # --enable_xformers_memory_efficient_attention \
