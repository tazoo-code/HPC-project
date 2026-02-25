#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --output=logs/03_inference_%A_%a.out
#SBATCH --error=logs/03_inference_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4    # 5 target ages: 10, 25, 40, 60, 80

# This job array runs inference for every target age on the entire val set.
# Each array task handles ONE target age — they all run in parallel.

set -euo pipefail

PROJECT_DIR="$HOME/age-diffusion"
SIF_PATH="$PROJECT_DIR/singularity/age_diffusion.sif"
LORA_WEIGHTS="$PROJECT_DIR/lora-age-finetuned"
VAL_IMAGES="$PROJECT_DIR/data/UTKFace_processed/val/images"
OUTPUT_BASE="$PROJECT_DIR/inference_results"

# Map SLURM_ARRAY_TASK_ID to a target age
TARGET_AGES=(10 25 40 60 80)
TARGET_AGE=${TARGET_AGES[$SLURM_ARRAY_TASK_ID]}

OUTPUT_DIR="$OUTPUT_BASE/age_${TARGET_AGE}"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_DIR/logs"

echo "========================================"
echo "Inference — Target Age: $TARGET_AGE"
echo "Array task:  $SLURM_ARRAY_TASK_ID / $SLURM_ARRAY_TASK_COUNT"
echo "Node:        $(hostname)"
echo "Started:     $(date)"
echo "========================================"

# Run inference on each image in the val set
for IMG in "$VAL_IMAGES"/*.jpg; do
    singularity exec \
        --nv \
        --bind "$PROJECT_DIR:$PROJECT_DIR" \
        "$SIF_PATH" \
        python "$PROJECT_DIR/scripts/inference.py" \
            --input_image   "$IMG" \
            --target_age    "$TARGET_AGE" \
            --lora_weights  "$LORA_WEIGHTS" \
            --output_dir    "$OUTPUT_DIR" \
            --strength      0.55 \
            --num_steps     30 \
            --seed          42
done

echo ""
echo "Inference complete for age $TARGET_AGE."
echo "Results in: $OUTPUT_DIR"
echo "Finished: $(date)"
