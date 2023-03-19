#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --mem=64000
#SBATCH --gres=gpu:A100m40:1
#SBATCH --constraint=A100
#SBATCH --job-name="mbert"
#SBATCH --output="mbert-output.out"
#SBATCH --mail-user=karolill@stud.ntnu.no
#SBATCH --mail-type=ALL

module purge
module load intel/2020b
module load Python/3.8.6-GCCcore-10.2.0
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1

python model_training_and_evaluation.py \
        --model_path 'bert-base-multilingual-cased' \
        --model_name 'mbert'