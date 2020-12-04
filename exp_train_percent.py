#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare training on different datasets

Execute training:
	$ python exp_train_percent.py ttv/ne_train_spa_2_na.tsv ttv/ne_test_spa_2_na.tsv -f tsv -act -past -pastact -rep
"""
import os
import sys
import random
import codecs
import argparse
import time, datetime
from collections import Counter
import json
import difflib

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import pycrfsuite

### Tag functions
from utils import dataset_labels
from crf_train import openData, add_feature_columns, get_features_from_row, word_bs_feature, generate_features
from crf_test import bio_classification_report, report_to_file, crf_predict

#### Read Data functions
def argparser():
	"""Creating arparse.ArgumentParser and returning arguments
	"""
	argparser = argparse.ArgumentParser(description='Train a CRF and test it.', formatter_class=argparse.RawTextHelpFormatter)
	# Data files
	argparser.add_argument('train', type=str, help="file listing train dialogs")
	argparser.add_argument('test', type=str, help="file listing train dialogs")
	argparser.add_argument('--format', '-f', choices=['txt', 'tsv'], required=True, help="data file format - all files must have the same format")
	argparser.add_argument('--txt_columns', nargs='+', type=str, default=[], help=""".txt columns name (in order); most basic txt is ['spa_all', 'ut', 'time', 'speaker', 'sentence']""")
	# Operations on data
	argparser.add_argument('--match_age', type=int, nargs='+', default=None, help="ages to match data to - for split analysis")
	argparser.add_argument('--keep_tag', choices=['all', '1', '2', '2a'], default="all", help="keep first part / second part / all tag")
	argparser.add_argument('--out', type=str, default='results', help="where to write .crfsuite model file")
	# parameters for training:
	argparser.add_argument('--nb_occurrences', '-noc', type=int, default=5, help="number of minimum occurrences for word to appear in features")
	argparser.add_argument('--use_action', '-act', action='store_true', help="whether to use action features to train the algorithm, if they are in the data")
	argparser.add_argument('--use_past', '-past', action='store_true', help="whether to add previous sentence as features")
	argparser.add_argument('--use_repetitions', '-rep', action='store_true', help="whether to check in data if words were repeated from previous sentence, to train the algorithm")
	argparser.add_argument('--use_past_actions', '-pa', action='store_true', help="whether to add actions from the previous sentence to features")
	argparser.add_argument('--train_percentage', type=float, nargs='+', default=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.], help="percentage (as fraction) of data to use for training. If --conv_split_length is not set, whole conversations will be used.")
	argparser.add_argument('--verbose', action="store_true", help="Whether to display training iterations output.")
	argparser.add_argument('--prediction_mode', choices=["raw", "exclude_ool"], default="exclude_ool", type=str, help="Whether to predict with NOL/NAT/NEE labels or not.")

	
	args = argparser.parse_args()
	return args

### REPORT
def plot_training(data, file_name):
	plt.figure()
	data.plot()
	plt.savefig(file_name+'.png')

#### MAIN
if __name__ == '__main__':
	args = argparser()
	print(args)

	# Definitions
	number_words_for_feature = args.nb_occurrences # default 5
	number_segments_length_feature = 10
	#number_segments_turn_position = 10 # not used for now
	training_tag = 'spa_'+args.keep_tag
	
	# Loading data
	if args.format == 'txt':
		if args.txt_columns == []:
			raise TypeError('--txt_columns [col0] [col1] ... is required with format txt')
		elif 'action' not in args.txt_columns:
			raise ValueError("in order to test the impact of actions, they must be in the data")
		# Loading with actions and repetitions in order to use them later
		data_train = openData(args.train, cut=args.cut, column_names=args.txt_columns, match_age=args.match_age, use_action = True, check_repetition=True, use_past=True, use_pastact=True)
		data_test = openData(args.test, cut=args.cut, column_names=args.txt_columns, match_age=args.match_age, use_action = True, check_repetition=True, use_past=True, use_pastact=True)

	elif args.format == 'tsv':
		data_train = pd.read_csv(args.train, sep='\t', keep_default_na=False).reset_index(drop=False)
		if 'action' not in data_train.columns.str.lower():
			raise ValueError("in order to test the impact of actions, they must be in the data")
		# Loading with actions and repetitions in order to use them later
		data_train.rename(columns={col:col.lower() for col in data_train.columns}, inplace=True)
		data_train = add_feature_columns(data_train, use_action=True, match_age=args.match_age, check_repetition=True, use_past=True, use_pastact=True)
		# Same for test
		data_test = pd.read_csv(args.test, sep='\t', keep_default_na=False).reset_index(drop=False)
		data_test.rename(columns={col:col.lower() for col in data_test.columns}, inplace=True)
		data_test = add_feature_columns(data_test, use_action=True, match_age=args.match_age, check_repetition=True, use_past=True, use_pastact=True)
		# Parameters
		training_tag = [x for x in data_train.columns if 'spa_' in x][0]
		args.training_tag = training_tag
	
	logger = {} # Dictionary containing results
	freport = {} # Dictionary containing reports
	name = os.path.join(os.getcwd(),('' if args.out is None else args.out), 
				'_'.join([ x for x in [os.path.basename(__file__).replace('.py',''), training_tag, datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')] if x ])) # Location for weight save
	os.mkdir(name)


	for percentage in args.train_percentage:
		# only take x % of the files
		train_files = data_train['file_id'].unique().tolist()
		train_subset = np.random.choice(len(train_files), size=int(len(train_files)*percentage), replace=False)
		train_files = [train_files[x] for x in train_subset]
		if len(train_files) == 0:
			continue
		else:
			print(f"\n### Training with {percentage*100}% - {len(train_files)} files: ")
			local_train = data_train[data_train['file_id'].isin(train_files)]
			nm = os.path.join(name, "{0:3.1f}%".format(percentage*100))

			# generating features
			features_idx = generate_features(local_train, training_tag, args.nb_occurrences, args.use_action, args.use_repetitions, bin_cut=number_segments_length_feature)

			# creating crf features set for train
			local_train['features'] = local_train.apply(lambda x: get_features_from_row(features_idx,
                                                                                        x.tokens, x['speaker'], x.turn_length,
                                                                                        action_tokens=None if not args.use_action else x.action_tokens,
                                                                                        repetitions=None if not args.use_repetitions else (x.repeated_words, x.nb_repwords, x.ratio_repwords),
                                                                                        past_tokens=None if not args.use_past else x.past,
                                                                                        pastact_tokens=None if not args.use_past_actions else x.past_act), axis=1)

			# Once the features are done, groupby name and extract a list of lists
			grouped_train = local_train.dropna(subset=[training_tag]).groupby(by=['file_id']).agg({
				'features' : lambda x: [y for y in x],
				training_tag : lambda x: [y for y in x], 
				'index': min
			}) # listed by apparition order
			grouped_train = sklearn.utils.shuffle(grouped_train)

			### Training
			trainer = pycrfsuite.Trainer(verbose=args.verbose)
			# Adding data
			for idx, file_data in grouped_train.iterrows():
				trainer.append(file_data['features'], file_data[training_tag]) # X_train, y_train
			# Parameters
			trainer.set_params({
					'c1': 1,   # coefficient for L1 penalty
					'c2': 1e-3,  # coefficient for L2 penalty
					'max_iterations': 50,  # stop earlier
					'feature.possible_transitions': True # include transitions that are possible, but not observed
			})
			print("Saving model at: {}".format(nm))
	
			trainer.train(nm +'_model.pycrfsuite')
			with open(nm+'_features.json', 'w') as json_file: # dumping features
				json.dump(features_idx, json_file)
			
			### Testing
			tagger = pycrfsuite.Tagger()
			tagger.open(nm +'_model.pycrfsuite')

			data_test['features'] = data_test.apply(lambda x: get_features_from_row(features_idx,
                                                                                    x.tokens, x['speaker'], x.turn_length,
                                                                                    action_tokens=None if not args.use_action else x.action_tokens,
                                                                                    repetitions=None if not args.use_repetitions else (x.repeated_words, x.nb_repwords, x.ratio_repwords),
                                                                                    past_tokens=None if not args.use_past else x.past,
                                                                                    pastact_tokens=None if not args.use_past_actions else x.past_act), axis=1)

			data_test.dropna(subset=[training_tag], inplace=True)
			X_dev = data_test.groupby(by=['file_id']).agg({ 
				'features' : lambda x: [y for y in x],
				'index': min
			})
			y_pred = crf_predict(tagger, X_dev.sort_values('index', ascending=True)['features'], mode=args.prediction_mode) 
			data_test['y_pred'] = [y for x in y_pred for y in x] # flatten
			data_test['y_true'] = data_test[training_tag]
			data_test['pred_OK'] = data_test.apply(lambda x: (x.y_pred == x.y_true), axis=1)
			# remove ['NOL', 'NAT', 'NEE'] for prediction and reports
			data_crf = data_test[~data_test['y_true'].isin(['NOL', 'NAT', 'NEE'])]
			# reports
			report, mat, acc, cks = bio_classification_report(data_crf['y_true'].tolist(), data_crf['y_pred'].tolist())
			logger[percentage] = acc
			freport[percentage] = {'report':report, 'cm':mat}

	train_per = pd.Series(logger, name='acc_over_train_percentage')

	report_to_file({ 
		**{ 'report_'+str(n) : d['report'].T for n, d in freport.items()},
		**{ 'cm_'+str(n) : d['cm'].T for n, d in freport.items()},
	}, os.path.join(name, 'report.xlsx'))

	with open(os.path.join(name, 'metadata.txt'), 'w') as meta_file: # dumping metadata
		for arg in vars(args):
			meta_file.write("{0}:\t{1}\n".format(arg, getattr(args, arg)))
		meta_file.write("{0}:\t{1}\n".format("Experiment", "Datasets"))
	
	# plotting training curves
	plot_training(train_per, os.path.join(name, 'percentage_evolution'))