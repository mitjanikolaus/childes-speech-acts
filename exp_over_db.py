#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare training on different datasets

Execute training:
	$ python exp_over_db.py ttv/childes_allannot_{}_spa_2.tsv ttv/childes_gael_{}_spa_2.tsv ttv/childes_naroll_{}_spa_2.tsv ttv/childes_ne_{}_spa_2.tsv -f tsv --use_action
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
	argparser.add_argument('data_pattern', type=str, nargs='+', help="patterns for the files listing dialogs - '{}' will be replaced with 'train'/'test'")
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
	argparser.add_argument('--verbose', action="store_true", help="Whether to display training iterations output.")
	argparser.add_argument('--prediction_mode', choices=["raw", "exclude_ool"], default="exclude_ool", type=str, help="Whether to predict with NOL/NAT/NEE labels or not.")
	
	args = argparser.parse_args()
	return args

def shorten_name(str1:str, str2:str) -> (str, str):
	"""Creating shorter names by looking up common patterns in naming and removing them.
	"""
	# looking up common patterns - focusing on patterns of length >= 3
	matches = difflib.SequenceMatcher(None, str1, str2).get_matching_blocks()
	# removing said patterns
	for x in [str1[match.a:(match.a+match.size)] for match in matches if match.size >= 3]:
		str1 = str1.replace(x, '')
	for x in [str2[match.b:(match.b+match.size)] for match in matches if match.size >= 3]:
		str2 = str2.replace(x, '')
	# return
	return str1, str2

#### MAIN
if __name__ == '__main__':
	args = argparser()
	print(args)

	# Definitions
	number_words_for_feature = args.nb_occurrences # default 5
	number_segments_length_feature = 10
	#number_segments_turn_position = 10 # not used for now
	training_tag = 'spa_'+args.keep_tag

	if args.format == 'txt':
		if args.txt_columns == []:
			raise TypeError('--txt_columns [col0] [col1] ... is required with format txt')
		args.use_action = args.use_action & ('action' in args.txt_columns)
		
		data_train = {
			db: openData(db.format('train'), column_names=args.txt_columns, match_age=args.match_age, use_action = args.use_action, check_repetition=args.use_repetitions) for db in args.data_pattern
		}
		data_test = {
			db: openData(db.format('test'), column_names=args.txt_columns, match_age=args.match_age, use_action = args.use_action, check_repetition=args.use_repetitions) for db in args.data_pattern
		}

	elif args.format == 'tsv':
		# Read data
		data_train = {
			db: pd.read_csv(db.format('train'), sep='\t', keep_default_na=False).reset_index(drop=False) for db in args.data_pattern
		}
		data_test = {
			db: pd.read_csv(db.format('test'), sep='\t', keep_default_na=False).reset_index(drop=False) for db in args.data_pattern
		}
		args.use_action = args.use_action & ('action' in list(data_train.values())[0].columns.str.lower())
		# Update data
		for db in args.data_pattern:
			data_train[db].rename(columns={col:col.lower() for col in data_train[db].columns}, inplace=True)
			data_train[db] = add_feature_columns(data_train[db], use_action=args.use_action, match_age=args.match_age, check_repetition=args.use_repetitions, use_past=args.use_past, use_pastact=args.use_past_actions)
			data_test[db].rename(columns={col:col.lower() for col in data_test[db].columns}, inplace=True)
			data_test[db] = add_feature_columns(data_test[db], use_action=args.use_action, match_age=args.match_age, check_repetition=args.use_repetitions, use_past=args.use_past, use_pastact=args.use_past_actions)
		training_tag = [x for x in list(data_train.values())[0].columns if 'spa_' in x][0]
		args.training_tag = training_tag
	
	logger = {db:{} for db in args.data_pattern} # Dictionary containing results
	freport = {} # Dictionary containing dataframes to save
	name = os.path.join(os.getcwd(),('' if args.out is None else args.out), 
				'_'.join([ x for x in [os.path.basename(__file__).replace('.py',''), training_tag, datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')] if x ])) # Location for weight save
	os.mkdir(name)
	# Training & Testing
	for db_train in args.data_pattern:
		nm = os.path.join(name, db_train.replace('/', '_'))
		# generating features
		features_idx = generate_features(data_train[db_train], training_tag, args.nb_occurrences, args.use_action, args.use_repetitions, bin_cut=number_segments_length_feature)

		# creating crf features set for train
		data_train[db_train]['features'] = data_train[db_train].apply(lambda x: get_features_from_row(features_idx,
                                                                                                      x.tokens, x['speaker'], x.turn_length,
                                                                                                      action_tokens=None if not args.use_action else x.action_tokens,
                                                                                                      repetitions=None if not args.use_repetitions else (x.repeated_words, x.nb_repwords, x.ratio_repwords),
                                                                                                      past_tokens=None if not args.use_past else x.past,
                                                                                                      pastact_tokens=None if not args.use_past_actions else x.past_act), axis=1)

		# Once the features are done, groupby name and extract a list of lists
		grouped_train = data_train[db_train].dropna(subset=[training_tag]).groupby(by=['file_id']).agg({
			'features' : lambda x: [y for y in x],
			training_tag : lambda x: [y for y in x], 
			'index': min
		}) # listed by apparition order
		grouped_train = sklearn.utils.shuffle(grouped_train)

		### Training
		print(f"\n### Training {db_train}: start.".upper())
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
		
		### Testing - looping over test samples
		tagger = pycrfsuite.Tagger()
		tagger.open(nm +'_model.pycrfsuite')

		for db_test in args.data_pattern: 
			data_test[db_test]['features'] = data_test[db_test].apply(lambda x: get_features_from_row(features_idx,
                                                                                                      x.tokens, x['speaker'], x.turn_length,
                                                                                                      action_tokens=None if not args.use_action else x.action_tokens,
                                                                                                      repetitions=None if not args.use_repetitions else (x.repeated_words, x.nb_repwords, x.ratio_repwords),
                                                                                                      past_tokens=None if not args.use_past else x.past,
                                                                                                      pastact_tokens=None if not args.use_past_actions else x.past_act), axis=1)

			data_test[db_test].dropna(subset=[training_tag], inplace=True)
			X_dev = data_test[db_test].groupby(by=['file_id']).agg({ 
				'features' : lambda x: [y for y in x],
				'index': min
			})
			y_pred = crf_predict(tagger, X_dev.sort_values('index', ascending=True)['features'], mode=args.prediction_mode) 
			data_test[db_test]['y_pred'] = [y for x in y_pred for y in x] # flatten
			data_test[db_test]['y_true'] = data_test[db_test][training_tag]
			data_test[db_test]['pred_OK'] = data_test[db_test].apply(lambda x: (x.y_pred == x.y_true), axis=1)
			# remove ['NOL', 'NAT', 'NEE'] for prediction and reports
			data_crf = data_test[db_test][~data_test[db_test]['y_true'].isin(['NOL', 'NAT', 'NEE'])]

			# reports
			report, mat, acc, cks = bio_classification_report(data_crf['y_true'].tolist(), data_crf['y_pred'].tolist())
			logger[db_train.format('train')][db_test.format('test')] = acc
			strain, stest = shorten_name(db_train, db_test)
			freport['{}>{}'.format(strain, stest)] = {'mat': mat, 'report': report}
	
	cross_training = pd.DataFrame(logger)

	report_to_file({ **{'cross_training': cross_training},
		**{ 'report_'+name : d['report'].T for name, d in freport.items()},
		**{ 'cm_'+name : d['mat'].T for name, d in freport.items()},
	}, os.path.join(name, 'report.xlsx'))

	with open(os.path.join(name, 'metadata.txt'), 'w') as meta_file: # dumping metadata
		for arg in vars(args):
			meta_file.write("{0}:\t{1}\n".format(arg, getattr(args, arg)))
		meta_file.write("{0}:\t{1}\n".format("Experiment", "Datasets"))