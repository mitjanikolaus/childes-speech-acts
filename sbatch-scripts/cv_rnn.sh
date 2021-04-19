#!/bin/bash
#
#SBATCH --job-name=cv_lstm
#
#SBATCH --ntasks=1
#SBATCH --time=90:00:00
#SBATCH --mem=16000
#SBATCH --output=out/cv_lstm.out
#SBATCH --error=out/cv_lstm.out

source activate questions-analysis
#python -u train.py --epochs 100 --lr 0.0001  --model lstm --save model_lstm_crf.pt --emsize 200 --nhid-words-lstm 200 --nhid-utterance-lstm 100
python -u nn_crossvalidation.py --data data/new_england_preprocessed.p --out lstm_cv/ --model lstm --epochs 50 

