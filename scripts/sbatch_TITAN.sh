#!/bin/bash

#SBATCH -o logs/%j_reproduction.out
#SBATCH -J NPU-NTU_Reproduction
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=12

#SBATCH --mem=64G
#SBATCH --partition=titan
#SBATCH --qos=titan
#SBATCH --nodelist titanrtx01

nvidia-smi

source ~/.bashrc
conda activate ano
cd ~/Projects/NPU-NTU_System_for_Voice_Privacy_2024_Challenge_Implementation

python train_TITAN.py