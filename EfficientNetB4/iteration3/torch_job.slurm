#!/bin/bash
# ====================================
#SBATCH --job-name=torch-garbage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32GB
#SBATCH --time=0-23:59:59
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
# ====================================

source ~/software/init-conda
conda activate pytorch

python food_main.py 
