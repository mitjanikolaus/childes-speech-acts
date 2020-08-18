#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Original code: https://github.com/scrapinghub/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb

Features originally include:
* 'PosTurnSeg<=i', 'PosTurnSeg>=i', 'PosTurnSeg=i' with i in 0, number_segments_turn_position
	* Removed: need adapted parameter, default parameter splits the text into 4 
* 'Length<-]', 'Length]->', 'Length[-]' with i in 0 .. inf with binning
	* Adapted: if need for *more* features, will be added
* 'Spk=='
	* Removed: no clue what this does
* 'Speaker' in ['CHI', 'MOM', etc]
	* kept but simplified

TODO:
* Wait _this_ yields a question: shouldn't we split files if no direct link between sentences? like activities changed
* Split trainings
* ADAPT TAGS: RENAME TAGS

COLUMN NAMES IN FILES:
FILE_ID SPA_X SPEAKER SENTENCE for tsv
SPA_ALL IT TIME SPEAKER SENTENCE for txt - then ACTION and PREV_SENTENCE ?


Execute training:
	$ python crf_test.py ttv/childes_ne_test_spa_2.tsv -f tsv -m results/spa_2_2020-08-17-133211
"""
import os
import sys
import random
import codecs
import argparse
import time, datetime
from collections import Counter
import json
import ast

import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite
from joblib import load

### Tag functions
from utils import dataset_labels
from crf_train import openData, data_add_features, word_to_feature, word_bs_feature


#### Read Data functions
def argparser():
	argparser = argparse.ArgumentParser(description='Train a CRF and test it.', formatter_class=argparse.RawTextHelpFormatter)
	# Data files
	argparser.add_argument('test', type=str, help="file listing test dialogs")
	argparser.add_argument('--format', '-f', choices=['txt', 'tsv'], required=True, help="data file format - adapt reading")
	argparser.add_argument('--txt_columns', nargs='+', type=str, default=[], help=""".txt columns name (in order); most basic txt is ['spa_all', 'ut', 'time', 'speaker', 'sentence']""")
	argparser.add_argument('--match_age', type=int, nargs='+', default=None, help="ages to match data to - for split analysis")
	argparser.add_argument('--model', '-m', required=True, type=str, default=None, help="folder containing model, features and metadata")
	# parameters for training/testing:
	argparser.add_argument('--col_ages', type=str, default=None, help="if not None, plot evolution of accuracy over age groups")

	args = argparser.parse_args()

	# Load training arguments
	text_file = open(os.path.join(args.model, 'metadata.txt'), "r")
	lines = text_file.readlines() # lines ending with "\n"
	for line in lines:
		arg_name, value = line[:-1].split(":\t")
		if arg_name not in ['format', 'txt_columns', 'match_age']: # don't replace existing arguments!
			try:
				setattr(args, arg_name, ast.literal_eval(value))
			except ValueError as e:
				if "malformed node or string" in str(e):
					setattr(args, arg_name, value)
					
	return args


#### Report functions
def features_report(tagg):
	""" Extracts weights and transitions learned from training
	From: https://github.com/scrapinghub/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb

	Input:
	--------
	tagg: pycrfsuite.Tagger

	Output:
	--------
	states: pd.DataFrame
	
	transitions: pd.DataFrame
	"""
	info = tagg.info()
	# Feature weights
	state_features = Counter(info.state_features)
	states = pd.DataFrame([{'weight':weight, 'label':label, 'attribute':attr} for (attr, label), weight in state_features.items()]).sort_values(by=['weight'], ascending=False)
	# Transitions
	trans_features = Counter(info.transitions)
	transitions = pd.DataFrame([{'label_from':label_from, 'label':label_to, 'likelihood':weight} for (label_from, label_to), weight in trans_features.items()]).sort_values(by=['likelihood'], ascending=False)
	# return
	return states, transitions

def bio_classification_report(y_true, y_pred):
	"""
	Classification report for a list of BIO-encoded sequences.
	It computes token-level metrics and discards "O" labels.
	Requires scikit-learn 0.20+ 

	Output:
	--------
	cr: pd.DataFrame

	cm: np.array

	acc: float
	"""	
	cr = classification_report(y_true, y_pred, digits = 3, output_dict=True)
	cm = confusion_matrix(y_true, y_pred, normalize='true')
	acc = sklearn.metrics.accuracy_score(y_true, y_pred, normalize = True)
	cks = cohen_kappa_score(y_true, y_pred)

	print("==> Accuracy: {0:.3f}".format(acc))
	print("==> Cohen Kappa Score: {0:.3f} \t(pure chance: {1:.3f})".format(cks, 1./len(set(y_true))))
	# using both as index in case not the same labels in it
	return pd.DataFrame(cr), pd.DataFrame(cm, index=sorted(set(y_true+y_pred)), columns=sorted(set(y_true+y_pred))), acc, cks

def plot_testing(test_df:pd.DataFrame, file_location:str, col_ages):
	"""Separating CHI/MOT and ages to plot accuracy, annotator agreement and number of categories over age.
	"""
	tmp = []
	speakers = test_df["speaker"].unique().tolist()
	for age in sorted(test_df[col_ages].unique().tolist()): # remove < 1Y?
		for spks in ([[x] for x in speakers]+[speakers]):
			age_loc_sub = test_df[(test_df[col_ages] == age) & (test_df.speaker.isin(spks))]
			acc = sklearn.metrics.accuracy_score(age_loc_sub.y_true, age_loc_sub.y_pred, normalize=True)
			cks = cohen_kappa_score(age_loc_sub.y_true, age_loc_sub.y_pred)
			tmp.append({'age':age, 'locutor':'&'.join(spks), 'accuracy':acc, 'agreement': cks, 'nb_labels': len(age_loc_sub.y_true.unique().tolist())})
		# also do CHI/MOT separately
	tmp = pd.DataFrame(tmp)
	speakers = tmp.locutor.unique().tolist()
	# plot
	fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(18,10))
	for i, col in enumerate(['accuracy', 'agreement', 'nb_labels']):
		for spks in speakers:
			ax[i].plot(tmp[tmp.locutor == spks].age, tmp[tmp.locutor == spks][col], label=spks)
			ax[i].set_ylabel(col)
	ax[2].set_xlabel('age (in months)')
	ax[2].legend()
	plt.savefig(file_location)

def report_to_file(dfs:dict, file_location:str):
	"""Looping on each pd.DataFrame to log to excel
	"""
	writer = pd.ExcelWriter(file_location)
	for name, data in dfs.items():
		data.to_excel(writer, sheet_name=name)
	writer.save()


#### MAIN
if __name__ == '__main__':
	args = argparser()

	# Definitions
	training_tag = args.training_tag

	if args.format == 'txt':
		if args.txt_columns == []:
			raise TypeError('--txt_columns [col0] [col1] ... is required with format txt')
		data_test = openData(args.test, column_names=args.txt_columns, match_age=args.match_age, use_action = args.use_action, check_repetition=args.use_repetitions)
	elif args.format == 'tsv':
		data_test = pd.read_csv(args.test, sep='\t').reset_index(drop=False)
		data_test.rename(columns={col:col.lower() for col in data_test.columns}, inplace=True)
		data_test['speaker'] = data_test['speaker'].apply(lambda x: x if x in ['CHI', 'MOT'] else 'MOT')
		data_test = data_add_features(data_test, use_action=args.use_action , match_age=args.match_age, check_repetition=args.use_repetitions)
	
	# Loading model
	name = args.model
	# loading features
	with open(os.path.join(name, 'features.json'), 'r') as json_file:
		features_idx = json.load(json_file)
	data_test['features'] = data_test.apply(lambda x: word_to_feature(features_idx, x.tokens, x['speaker'], x.turn_length, None if not args.use_action else x.action_tokens, None if not args.use_repetitions else (x.repeated_words, x.nb_repwords, x.ratio_repwords)), axis=1)

	# Predictions
	tagger = pycrfsuite.Tagger()
	tagger.open(os.path.join(name,'model.pycrfsuite'))
	
	# creating data - TODO: dropna???
	data_test.dropna(subset=[training_tag], inplace=True)
	X_dev = data_test.groupby(by=['file_id']).agg({ 
		'features' : lambda x: [y for y in x],
		'index': min
	})
	y_pred = [tagger.tag(xseq) for xseq in X_dev.sort_values('index', ascending=True)['features']]
	data_test['y_pred'] = [y for x in y_pred for y in x] # flatten
	data_test['y_true'] = data_test[training_tag]
	data_test['pred_OK'] = data_test.apply(lambda x: (x.y_pred == x.y_true), axis=1)
	# reports
	report, mat, acc, cks = bio_classification_report(data_test['y_true'].tolist(), data_test['y_pred'].tolist())
	states, transitions = features_report(tagger)
	
	int_cols = ['file_id', 'speaker'] + ([args.col_ages] if args.col_ages is not None else []) + [x for x in data_test.columns if 'spa_' in x] + ['y_true', 'y_pred', 'pred_OK']

	report_to_file({
		'test_data': data_test[int_cols],
		'classification_report': report.T,
		'confusion_matrix': mat,
		'weights': states, 
		'learned_transitions': transitions.pivot(index='label_from', columns='label', values='likelihood') 
	}, os.path.join(name, args.test.replace('/', '_')+'_report.xlsx'))

	if args.col_ages is not None:
		plot_testing(data_test, os.path.join(name, args.test+'_agesevol.png'), args.col_ages)

	# Test baseline
	if args.baseline is not None:
		bs_model = load(os.path.join(name, 'baseline.joblib'))

		print("\nBaseline model for comparison:")
		X = data_test.dropna(subset=[training_tag]).apply(lambda x: word_bs_feature(features_idx, x.tokens, x['speaker'], x.turn_length, None if not args.use_action else x.action_tokens, None if not args.use_repetitions else (x.repeated_words, x.nb_repwords, x.ratio_repwords)), axis=1)
		y = data_test.dropna(subset=[training_tag])[training_tag].tolist()
		# ID from label - bidict
		labels = dataset_labels(training_tag.upper())
		# transforming
		X = np.array(X.tolist())
		y = np.array([labels[lab] for lab in y]) # to ID

		y_pred = bs_model.predict(X)
		_ = bio_classification_report(y.tolist(), y_pred.tolist())
