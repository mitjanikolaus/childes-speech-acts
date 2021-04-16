#!/bin/bash
#
#SBATCH --job-name=train
#
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem=16000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/train_rnn.out
#SBATCH --error=out/train_rnn.out

source activate questions-analysis
python -u nn_train.py --data data/new_england_preprocessed.p --out lstm_pos/ --model lstm --epochs 100
