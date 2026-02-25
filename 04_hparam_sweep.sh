#!/bin/bash
#SBATCH --job-name=hparam_sweep
#SBATCH --output=logs/04_sweep_%A_%a.out
#SBATCH --error=logs/04_sweep_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --array=0-5   # 6 combinations

# Hyperparameter sweep: 3 LoRA ranks x 2 learning rates = 6 jobs
# Each job is independent and runs in parallel — classic HPC job array usage.

set -euo pipefail

PROJECT_DIR="$HOME/age-diffusion"
SIF_PATH="$PROJECT_DIR/singularity/age_diffusion.sif"
PROCESSED_DIR="$PROJECT_DIR/data/UTKFace_processed"

NUM_GPUS=2

mkdir -p "$PROJECT_DIR/logs"

# Define the sweep grid
LORA_RANKS=(4 4 8 8 16 16)
LEARNING_RATES=(5e-5 1e-4 5e-5 1e-4 5e-5 1e-4)

RANK=${LORA_RANKS[$SLURM_ARRAY_TASK_ID]}
LR=${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}
RUN_NAME="rank${RANK}_lr${LR}"
OUTPUT_DIR="$PROJECT_DIR/sweep/${RUN_NAME}"

mkdir -p "$OUTPUT_DIR"

ACCEL_CONFIG="$PROJECT_DIR/accelerate_sweep_${SLURM_ARRAY_TASK_ID}.yaml"
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

echo "========================================"
echo "Hyperparameter Sweep"
echo "Task ID:    $SLURM_ARRAY_TASK_ID"
echo "LoRA rank:  $RANK"
echo "LR:         $LR"
echo "Output:     $OUTPUT_DIR"
echo "Node:       $(hostname)"
echo "Started:    $(date)"
echo "========================================"

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
            --max_train_steps=5000 \
            --learning_rate="$LR" \
            --lr_scheduler="cosine" \
            --lr_warmup_steps=200 \
            --rank="$RANK" \
            --mixed_precision="fp16" \
            --checkpointing_steps=5000 \
            --output_dir="$OUTPUT_DIR" \
            --seed=42 \
            --report_to="tensorboard"

echo ""
echo "Sweep run complete: $RUN_NAME"
echo "Finished: $(date)"

# Clean up tmp accelerate config
rm -f "$ACCEL_CONFIG"
