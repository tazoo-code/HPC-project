#!/bin/bash
#SBATCH --job-name=lora_train
#SBATCH --output=logs/02_train_%j.out
#SBATCH --error=logs/02_train_%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:4
# Adjust --gres and --partition to match your cluster.
# Common: gpu:2, gpu:4, gpu:a100:2, etc.

set -euo pipefail

PROJECT_DIR="$HOME/age-diffusion"
SIF_PATH="$PROJECT_DIR/singularity/age_diffusion.sif"
PROCESSED_DIR="$PROJECT_DIR/data/UTKFace_processed"
OUTPUT_DIR="$PROJECT_DIR/lora-age-finetuned"

NUM_GPUS=4   # must match --gres above

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "LoRA Fine-tuning"
echo "Node:     $(hostname)"
echo "GPUs:     $NUM_GPUS"
echo "Started:  $(date)"
echo "========================================"

# Write accelerate config inside the container's writable tmp
ACCEL_CONFIG="$PROJECT_DIR/accelerate_config.yaml"
cat > "$ACCEL_CONFIG" <<EOF
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_machines: 1
num_processes: $NUM_GPUS
mixed_precision: fp16
gpu_ids: all
main_training_function: main
downcast_bf16: 'no'
EOF

singularity exec \
    --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    "$SIF_PATH" \
    accelerate launch \
        --config_file "$ACCEL_CONFIG" \
        --num_processes "$NUM_GPUS" \
        /opt/diffusers_scripts/train_text_to_image_lora.py \
            --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
            --train_data_dir="$PROCESSED_DIR/train" \
            --caption_column="text" \
            --resolution=512 \
            --center_crop \
            --random_flip \
            --train_batch_size=4 \
            --gradient_accumulation_steps=4 \
            --gradient_checkpointing \
            --max_train_steps=10000 \
            --learning_rate=1e-4 \
            --lr_scheduler="cosine" \
            --lr_warmup_steps=500 \
            --rank=8 \
            --mixed_precision="fp16" \
            --checkpointing_steps=2000 \
            --checkpoints_total_limit=3 \
            --output_dir="$OUTPUT_DIR" \
            --seed=42 \
            --report_to="tensorboard"

echo ""
echo "Training complete. Weights saved to: $OUTPUT_DIR"
echo "Finished: $(date)"
