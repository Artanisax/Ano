#!/bin/bash

# option gpu=* --nodelist ailab01 -p L40 --gres=gpu::$0 --time 2-00:00:00 --qos normal # L40 on ailab01
# option gpu=* --nodelist ailab01 -p L40 --gres=gpu:rtx5880ada:$0 --time 2-00:00:00 --qos normal # rtx5880ada on ailab01
# option gpu=* --nodelist ailab02 -p RTX5880Ada --gres=gpu:rtx5880ada:$0 --time 2-00:00:00 --qos normal # RTX5880Ada on ailab02
# option gpu=* --nodelist ailab03 -p RTXPRO5000 --gres=gpu:rtx5880ada:$0 --time 2-00:00:00 --qos normal # RTX5880Ada on ailab03
# option gpu=* --nodelist ailab03 -p RTXPRO5000 --gres=gpu:rtxpro5000:$0 --time 2-00:00:00 --qos normal # RTXPRO5000 on ailab03
# option gpu=* --nodelist ailab04 -p RTXPRO5000 --gres=gpu:rtxpro5000:$0 --time 2-00:00:00 --qos normal # RTXPRO5000 on ailab04

#SBATCH -o logs/%j_reproduction.out
#SBATCH -J CKR_Reproduction
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20

#SBATCH --nodelist=ailab02
#SBATCH --partition=RTX5880Ada
#SBATCH --gres=gpu:rtx5880ada:1
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --qos normal

source ~/.bashrc
conda activate ano_old
cd /data1/cse12110524/NPU-NTU_System_for_Voice_Privacy_2024_Challenge_Implementation

python train.py