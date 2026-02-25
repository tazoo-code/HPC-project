#!/bin/bash
#SBATCH --job-name=build_sif
#SBATCH --output=logs/00_build_%j.out
#SBATCH --error=logs/00_build_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# NOTE: building Singularity containers usually requires a non-GPU node
# and may need --partition=build or similar depending on your cluster.
# Check with your HPC admin if unsure.

set -euo pipefail

PROJECT_DIR="$HOME/age-diffusion"
SIF_PATH="$PROJECT_DIR/singularity/age_diffusion.sif"
DEF_PATH="$PROJECT_DIR/singularity/age_diffusion.def"

mkdir -p "$PROJECT_DIR/logs"

echo "========================================"
echo "Building Singularity container"
echo "Node:    $(hostname)"
echo "Started: $(date)"
echo "========================================"

# Some clusters require SINGULARITY_TMPDIR to point to a large scratch space
# export SINGULARITY_TMPDIR=$SCRATCH/tmp
# mkdir -p $SINGULARITY_TMPDIR

singularity build \
    --force \
    "$SIF_PATH" \
    "$DEF_PATH"

echo ""
echo "Build complete: $SIF_PATH"
echo "Size: $(du -sh $SIF_PATH | cut -f1)"
echo "Finished: $(date)"
