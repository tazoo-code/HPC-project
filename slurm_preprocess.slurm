#!/bin/bash

#SBATCH -J preprocess
#SBATCH -o logs/output_%j.txt
#SBATCH -e logs/errors_%j.txt

#SBATCH -p allgroups
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -c 6
#SBATCH --mem 32G

python preprocess.py
