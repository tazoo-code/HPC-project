#!/bin/bash

model=${1:-"stable-diffusion-v1-5/stable-diffusion-v1-5"}  # which diffusion model to fine tune
batch_size=${2:-16}  # per-GPU batch size (adjust based on your GPU memory)
rank=${3:-8}  # LoRA rank

num_gpus=$(nvidia-smi -L | wc -l)

ACCEL_CONFIG="accelerate_config.yaml"
cat > "$ACCEL_CONFIG" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: $num_gpus
mixed_precision: fp16
gpu_ids: all
main_training_function: main
downcast_bf16: 'no'
EOF

accelerate launch \
    --config_file "$ACCEL_CONFIG" \
    --num_processes "$num_gpus" \
    train_text_to_image_lora.py \
        --pretrained_model_name_or_path="$model" \
        --train_data_dir="data/UTKFace_processed" \
        --caption_column="text" \
        --resolution=512 \
        --center_crop \
        --random_flip \
        --train_batch_size=$batch_size \
        --gradient_accumulation_steps=$((16 / $batch_size)) \
        --gradient_checkpointing \
        --max_train_steps=10000 \
        --learning_rate=1e-4 \
        --lr_scheduler="cosine" \
        --lr_warmup_steps=500 \
        --rank=${rank} \
        --mixed_precision="fp16" \
        --checkpointing_steps=1000 \
        --checkpoints_total_limit=3 \
        --output_dir="lora-full-run" \
        --seed=42 \
        --report_to="wandb" \
        --validation_prompt="a portrait photo of a DESC, aged AGE, realistic, high quality" \
        --num_validation_images=8