#!/bin/bash
#
#SBATCH --job-name=cv_bert
#
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --mem=16000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out/cv_transformer_lstm_100.out
#SBATCH --error=out/cv_transformer_lstm_100.out

source activate speech-acts
python -u nn_crossvalidation.py --data data/new_england_preprocessed.p --out transformer_lstm_100/ --model transformer --epochs 100 --lr 0.00001

