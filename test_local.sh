#!/bin/bash
# test_local.sh - Full end-to-end local run for the age progression pipeline.
#
# Usage:
#   chmod +x test_local.sh
#   ./test_local.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="hpc1"
DATA_DIR="$PROJECT_DIR/dataset/UTKFace"
PROCESSED_DIR="$PROJECT_DIR/data/UTKFace_processed_test"
LORA_OUT="$PROJECT_DIR/lora-full-run"
INFERENCE_OUT="$PROJECT_DIR/inference-results"
INFERENCE_SCRIPT="$PROJECT_DIR/inference.py"
TRAIN_SCRIPT="$PROJECT_DIR/train_text_to_image_lora.py"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
step() { echo -e "\n${GREEN}> $1${NC}"; }
warn() { echo -e "${YELLOW}!  $1${NC}"; }

# ---------------------------------------------------------------------------
# STEP 0 -- Activate conda environment
# ---------------------------------------------------------------------------
step "Step 0/4 -- Activating conda environment '$CONDA_ENV'"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "Python : $(which python)"
echo "Torch  : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA   : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU    : $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

# ---------------------------------------------------------------------------
# STEP 1 -- Download LoRA training script (if needed)
# ---------------------------------------------------------------------------
step "Step 1/5 -- Fetching LoRA training script"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    wget -q \
        https://raw.githubusercontent.com/huggingface/diffusers/v0.27.0/examples/text_to_image/train_text_to_image_lora.py \
        -O "$TRAIN_SCRIPT"
    echo "Downloaded: $TRAIN_SCRIPT"
else
    echo "Training script already present, skipping download."
fi

# ---------------------------------------------------------------------------
# STEP 2 -- Preprocess UTKFace
# ---------------------------------------------------------------------------
step "Step 2/5 -- Preprocessing dataset"

if [ ! -d "$DATA_DIR" ]; then
    warn "No UTKFace images found at $DATA_DIR"
    echo ""
    echo "Please download UTKFace and place the .jpg files in: $DATA_DIR"
    exit 1
fi

mkdir -p "$PROCESSED_DIR"

python "$PROJECT_DIR/preprocess.py" \
    --data_dir   "$DATA_DIR" \
    --output_dir "$PROCESSED_DIR" \
    --image_size 512 \
    --val_split  0.2 \
    --max_age    99 \
    --seed       42

echo "Train: $(find "$PROCESSED_DIR/train/images" -name '*.jpg' | wc -l) images"
echo "Val  : $(find "$PROCESSED_DIR/val/images"   -name '*.jpg' | wc -l) images"

# ---------------------------------------------------------------------------
# STEP 3 -- Full training (5000 steps, 1 GPU, ~3-4h)
# ---------------------------------------------------------------------------
step "Step 3/5 -- Full LoRA training (5000 steps)"
warn "Expected time: ~3-4 hours on a single GPU."
warn "Checkpoints saved every 1000 steps to $LORA_OUT -- safe to resume if interrupted."
echo "Training data: $(find "$PROCESSED_DIR/train/images" -name '*.jpg' | wc -l) images"

mkdir -p "$LORA_OUT"

accelerate launch \
    --num_processes 1 \
    --mixed_precision fp16 \
    "$TRAIN_SCRIPT" \
        --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
        --train_data_dir="$PROCESSED_DIR/train" \
        --caption_column="text" \
        --resolution=512 \
        --center_crop \
        --random_flip \
        --train_batch_size=4 \
        --gradient_accumulation_steps=2 \
        --gradient_checkpointing \
        --max_train_steps=5000 \
        --learning_rate=1e-4 \
        --lr_scheduler="cosine" \
        --lr_warmup_steps=200 \
        --rank=8 \
        --mixed_precision="fp16" \
        --checkpointing_steps=1000 \
        --checkpoints_total_limit=3 \
        --output_dir="$LORA_OUT" \
        --seed=42 \
        --report_to="tensorboard"

echo "LoRA weights saved to: $LORA_OUT"

# ---------------------------------------------------------------------------
# STEP 4 -- Inference at five target ages
# ---------------------------------------------------------------------------
step "Step 4/5 -- Inference (one image -> ages 10, 25, 40, 60, 80)"

mkdir -p "$INFERENCE_OUT"

set +e
TEST_IMAGE=$(find "$PROCESSED_DIR/val/images" -name "*.jpg" -type f | sort | head -1)
echo "Test image: $TEST_IMAGE"

if [ -z "$TEST_IMAGE" ]; then
    echo "ERROR: No val images found in $PROCESSED_DIR/val/images"
    exit 1
fi

for AGE in 10 25 40 60 80; do
    echo "  -> Generating age $AGE..."
    python "$INFERENCE_SCRIPT" \
        --input_image  "$TEST_IMAGE" \
        --target_age   "$AGE" \
        --lora_weights "$LORA_OUT" \
        --output_dir   "$INFERENCE_OUT" \
        --strength     0.55 \
        --num_steps    30 \
        --seed         42
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "  FAILED for age $AGE (exit code $STATUS)"
    else
        echo "  OK: age $AGE done"
    fi
done
set -e

# ---------------------------------------------------------------------------
# STEP 5 -- Summary
# ---------------------------------------------------------------------------
step "Step 5/5 -- Done!"

echo ""
echo "  Training data  : $PROCESSED_DIR"
echo "  LoRA weights   : $LORA_OUT"
echo "  Inference outs : $INFERENCE_OUT"
echo ""
echo "Generated images:"
ls "$INFERENCE_OUT"/*.jpg 2>/dev/null | sed 's/^/    /'
echo ""
echo "Next steps:"
echo "  * View results in $INFERENCE_OUT"
echo "  * Run the full multi-GPU training on the cluster: sbatch slurm/02_train.sh"