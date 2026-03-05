#!/bin/bash

#SBATCH -J train
#SBATCH -o logs/output_%j.txt
#SBATCH -e logs/errors_%j.txt

#SBATCH -p allgroups
#SBATCH -n 1
#SBATCH -t 12:00:00
#SBATCH -c 6
#SBATCH --mem 32G
#SBATCH --gres=gpu:rtx:2

./train.sh "stable-diffusion-v1-5/stable-diffusion-v1-5" 16 8
python inference.py
