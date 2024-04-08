#!/bin/bash
# ==============================================
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
# ===============================================

source ~/software/init-conda

conda activate pytorch

python ~/food_1_rn50_main.py --best_model_path ~/bestmodel_1_rn50.pth --images_path ~/food-101/ --transfer_learning True

conda deactivate
