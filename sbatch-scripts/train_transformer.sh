#!/bin/bash
#
#SBATCH --job-name=train
#
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem=16000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/train_transformer.out
#SBATCH --error=out/train_transformer.out

source activate questions-analysis
python -u train.py --epochs 20 --model transformer --lr 0.00001 --save model_transformer.pt --context 3
