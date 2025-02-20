#!/bin/bash
#SBATCH --gres=gpu:4 --constraint='gpu_mem:48GB'
#SBATCH --cpus-per-task=1
#SBATCH --mem=300G
#SBATCH --time=48:00:00
#SBATCH --job-name=gnnpr2
#SBATCH --partition=medium
module load Anaconda3
source activate $DATA/durga
python tryme.py


