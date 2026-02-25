#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --output=logs/01_preprocess_%j.out
#SBATCH --error=logs/01_preprocess_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
# No GPU needed for preprocessing

set -euo pipefail

PROJECT_DIR="$HOME/age-diffusion"
SIF_PATH="$PROJECT_DIR/singularity/age_diffusion.sif"

# ── Paths — adjust to your cluster's filesystem ───────────────────────────────
# Raw UTKFace images (download manually and place here, or use $SCRATCH)
RAW_DATA_DIR="$PROJECT_DIR/data/UTKFace_raw"

# Processed output (use a fast shared filesystem if available)
PROCESSED_DIR="$PROJECT_DIR/data/UTKFace_processed"

mkdir -p "$PROJECT_DIR/logs"

echo "========================================"
echo "UTKFace Preprocessing"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "Source:  $RAW_DATA_DIR"
echo "Output:  $PROCESSED_DIR"
echo "========================================"

singularity exec \
    --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    "$SIF_PATH" \
    python "$PROJECT_DIR/scripts/preprocess.py" \
        --data_dir    "$RAW_DATA_DIR" \
        --output_dir  "$PROCESSED_DIR" \
        --image_size  512 \
        --val_split   0.05 \
        --max_age     99 \
        --seed        42

echo ""
echo "Preprocessing complete."
echo "Train images: $(find $PROCESSED_DIR/train/images -name '*.jpg' | wc -l)"
echo "Val images:   $(find $PROCESSED_DIR/val/images   -name '*.jpg' | wc -l)"
echo "Finished: $(date)"
