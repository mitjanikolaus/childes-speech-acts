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
SPA_ALL IT TIME SPEAKER SENTENCE for txt - then ACTION

Execute training:
	$ python crf_train.py ttv/childes_ne_train_spa_2.tsv -act  -f tsv
"""
import os
import sys
import random
import codecs
import argparse
import time, datetime
from collections import Counter
import json
from typing import Union, Tuple

import re

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm, naive_bayes, ensemble
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite
from joblib import dump

### Tag functions
from utils import dataset_labels, select_tag


#### Read Data functions
def argparser():
	"""Creating arparse.ArgumentParser and returning arguments
	"""
	argparser = argparse.ArgumentParser(description='Train a CRF and test it.', formatter_class=argparse.RawTextHelpFormatter)
	# Data files
	argparser.add_argument('train', type=str, help="file listing train dialogs")
	argparser.add_argument('--format', '-f', choices=['txt', 'tsv'], required=True, help="data file format - adapt reading")
	argparser.add_argument('--txt_columns', nargs='+', type=str, default=[], help=""".txt columns name (in order); most basic txt is ['spa_all', 'ut', 'time', 'speaker', 'sentence']""")
	# Operations on data
	argparser.add_argument('--match_age', type=int, nargs='+', default=None, help="ages to match data to - for split analysis")
	argparser.add_argument('--keep_tag', choices=['all', '1', '2', '2a'], default="all", help="keep first part / second part / all tag")
	argparser.add_argument('--cut', type=int, default=1000000, help="if specified, use the first n train dialogs instead of all.")
	argparser.add_argument('--out', type=str, default='results', help="where to write .crfsuite model file")
	# parameters for training:
	argparser.add_argument('--nb_occurrences', '-noc', type=int, default=5, help="number of minimum occurrences for word to appear in features")
	argparser.add_argument('--use_action', '-act', action='store_true', help="whether to use action features to train the algorithm, if they are in the data")
	argparser.add_argument('--use_past', '-past', action='store_true', help="whether to add previous sentence as features")
	argparser.add_argument('--use_repetitions', '-rep', action='store_true', help="whether to check in data if words were repeated from previous sentence, to train the algorithm")
	argparser.add_argument('--use_past_actions', '-pa', action='store_true', help="whether to add actions from the previous sentence to features")
	argparser.add_argument('--train_percentage', type=float, default=1., help="percentage (as fraction) of data to use for training. If --conv_split_length is not set, whole conversations will be used.")
	argparser.add_argument('--verbose', action="store_true", help="Whether to display training iterations output.")
	# Baseline model
	argparser.add_argument('--baseline', type=str, choices=['SVC','LSVC', 'NB', 'RF'], default=None, help="which algorithm to use for baseline: SVM (classifier ou linear classifier), NaiveBayes, RandomForest(100 trees)")
	argparser.add_argument('--balance_ex', action="store_true", help="whether to take proportion of each class into account when training (imbalanced dataset).")

	args = argparser.parse_args()
	if (args.train_percentage is not None) and (args.train_percentage <=0. and args.train_percentage > 1.):
		raise ValueError("--train_percentage must be between 0. and 1. (strictly superior to 0, otherwise no data to train on). Current value: {0}.".format(args.train_percentage))

	return args


def openData(list_file:str, cut:int=100000, column_names:list=['all', 'ut', 'time', 'speaker', 'sentence'], **kwargs):
	"""
	Input:
	------
	list_file: `str`
		location of file containing train/dev/test txt files

	cut: `int`
		number of files to keep
	
	column_names: `list`
		list of features in the text file
	
	Kwargs:
	-------
	match_age: `Union[str,list]`
		list of ages to match column age_months to - if needed by later analysis. Matching column to closest value in list.
	
	use_action: `bool`
		whether to add actions to features
	
	use_past: `bool`
		whether to add previous sentence (full, unlike repetitions) to features
	
	use_pastact: `bool`
		whether to add actions to features
	
	check_repetition: `bool`
		whether to add repetition features

	Output:
	------
	p: `pd.DataFrame`
	"""
	print("Loading ", list_file)
	text_file = open(list_file, "r")
	lines = text_file.readlines() # lines ending with "\n"
	text_file.close()
	# loading data
	p = []
	for i in range(min(len(lines), cut)):
		file_name = lines[i][:-1]
		tmp = pd.read_csv(file_name, sep="\t", names=column_names, keep_default_na=False)
		tmp['file_id'] = file_name
		tmp['index'] = i
		p.append(tmp)
	p = pd.concat(p)
	# Changing locutors: INV/FAT become mother
	p['speaker'] = p['speaker'].apply(lambda x: x if x in ['CHI', 'MOT'] else 'MOT')
	# Adding features
	p = data_add_features(p, **kwargs)
	# Splitting tags
	for col_name, t in zip(['spa_1', 'spa_2', 'spa_2a'], ['first', 'second', 'adapt_second']):
		p[col_name] = p['spa_all'].apply(lambda x: select_tag(x, keep_part=t)) # creating columns with different tags
	# Return
	return p

#### Features functions
def data_add_features(p:pd.DataFrame, match_age:Union[str,list] = None, 
						use_action:bool = False, use_past:bool = False, use_pastact:bool = False, 
						check_repetition:bool = False):
	"""Function adding features to the data:
		* tokens: splitting spoken sentence into individual words
		* turn_length
		* tags (if necessary): extract interchange/illocutionary from general tag
		* action_tokens (if necessary): splitting action sentence into individual words
		* age_months: matching age to experimental labels
		* repeted_words:
		* number of repeated words
		* ratio of words that were repeated from previous sentence over sentence length
	"""
	# sentence: using tokens to count & all
	p['tokens'] = p.sentence.apply(lambda x: nltk.word_tokenize(x))
	p['turn_length'] = p.tokens.apply(len)
	# action: creating action tokens
	if use_action:
		p['action'].fillna('', inplace=True)
		p['action_tokens'] = p.action.apply(lambda x: x.lower().split())
	# matching age with theoretical age from the study
	# p['age_months'] = p.file.apply(lambda x: int(x.split('/')[-2])) # NewEngland only
	if 'age_months' in p.columns and match_age is not None:
		match_age = match_age if isinstance(match_age, list) else [match_age]
		p['age_months'] = p.age_months.apply(lambda age: min(match_age, key=lambda x:abs(x-age)))
	# repetition features
	if check_repetition or use_past or use_pastact:
		p['prev_file'] = p.file_id.shift(1).fillna(p.file_id.iloc[0])
		p['prev_spk'] = p.speaker.shift(1).fillna(p.speaker.iloc[0])
		p['prev_st'] = p.tokens.shift(1)#.fillna(p.tokens.iloc[0]) # doesn't work - fillna doesn't accept a list as value
		p['prev_st'].iloc[0] = p.tokens.iloc[0]
	if check_repetition:
		p['repeated_words'] = p.apply(lambda x: [w for w in x.tokens if w in x.prev_st] if (x.prev_spk != x.speaker) and (x.file_id == x.prev_file) else [], axis=1)
		p['nb_repwords'] = p.repeated_words.apply(len)
		p['ratio_repwords'] = p.nb_repwords/p.turn_length
	if use_past:
		p['past'] = p.apply(lambda x: x.prev_st if (x.file_id == x.prev_file) else [], axis=1)
	if use_action and use_pastact:
		p['prev_act'] = p['action_tokens'].shift(1)
		p['prev_act'].iloc[0] = p['action_tokens'].iloc[0]
		p['past_act'] = p.apply(lambda x: x.prev_act if (x.file_id == x.prev_file) else [], axis=1)
	# remove useless columns
	p = p[[col for col in p.columns if col not in ['prev_spk', 'prev_st', 'prev_file', 'prev_act']]]
	# return Dataframe
	return p


def word_to_feature(features:dict, spoken_tokens:list, speaker:str, ln:int, **kwargs):
	"""Replacing input list tokens with feature index

	Features should be of type:
	https://python-crfsuite.readthedocs.io/en/latest/pycrfsuite.html#pycrfsuite.ItemSequence
	==> Using Counters

	Input:
	-------
	features: `dict`
		dictionary of all features used, by type: {'words':Counter(), ...}

	spoken_tokens: `list`
		data sentence
	
	speaker: `str`
		MOT/CHI
	
	ln: `int`
		sentence length

	Kwargs:
	--------
	action_tokens: `list`
		data action if actions are not taken into account
	
	past_tokens: `list`

	pastact_tokens: `list`

	repetitions: `Tuple[list, float, float]`
		contains the list of repeated words, number of words repeated, ratio of repeated words over sequence
	
	Output:
	-------
	feat_glob: `dict`
		dictionary of same shape as feature, but only containing features relevant to data line
	"""
	feat_glob = { 'words': Counter([w for w in spoken_tokens if (w in features['words'].keys())]) } # TODO: add 'UNK' token
	feat_glob['speaker'] = {speaker:1.0}
	feat_glob['length'] = {k:(1 if ln <= float(k.split('-')[1]) and ln >= float(k.split('-')[0]) else 0) for k in features['length_bins'].keys()}

	if ('action_tokens' in kwargs) and (kwargs['action_tokens'] is not None):
		# actions are descriptions just like 'words'
		feat_glob['actions'] = Counter([w for w in kwargs['action_tokens'] if (w in features['action'].keys())]) #if (features['action'] is not None) else Counter(action_tokens)
	if ('repetitions' in kwargs) and (kwargs['repetitions'] is not None):
		(rep_words, len_rep, ratio_rep) = kwargs['repetitions']
		feat_glob['repeated_words'] = Counter([w for w in rep_words if (w in features['words'].keys())])
		feat_glob['rep_length'] = {k:(1 if len_rep <= float(k.split('-')[1]) and len_rep >= float(k.split('-')[0]) else 0) for k in features['rep_length_bins'].keys()}
		feat_glob['rep_ratio'] = {k:(1 if ratio_rep <= float(k.split('-')[1]) and ratio_rep >= float(k.split('-')[0]) else 0) for k in features['rep_ratio_bins'].keys()}
	if ('past_tokens' in kwargs) and (kwargs['past_tokens'] is not None):
		feat_glob['past'] = Counter([w for w in kwargs['past_tokens'] if (w in features['words'].keys())])
	if ('pastact_tokens' in kwargs) and (kwargs['pastact_tokens'] is not None):
		feat_glob['past_actions'] = Counter([w for w in kwargs['pastact_tokens'] if (w in features['action'].keys())])

	return feat_glob


def word_bs_feature(features:dict, spoken_tokens:list, speaker:str, ln:int, **kwargs):
	"""Replacing input list tokens with feature index

	Input:
	-------
	features: `dict`
		dictionary of all features used, by type: {'words':Counter(), ...}

	spoken_tokens: `list`
		data sentence
	
	speaker: `str`
		MOT/CHI
	
	ln: `int`
		sentence length
	
	Kwargs:
	-------
	action_tokens: `list`
		data action, default None if actions are not taken into account
	
	repetitions: `Tuple[list, float, float]`
		contains the list of repeated words, number of words repeated, ratio of repeated words over sequence
	
	Output:
	-------
	features_glob: `list`
		list of size nb_features, dummy of whether feature is contained or not
	"""
	nb_features = max([max([int(x) for x in v.values()]) for v in features.values()])+1
	# list features
	features_sparse = [features['words'][w] for w in spoken_tokens if w in features['words'].keys()] # words
	features_sparse.append(features['speaker'][speaker]) # locutor
	for k in features['length_bins'].keys(): # length
		if ln <= float(k.split('-')[1]) and ln >= float(k.split('-')[0]):
			features_sparse.append(features['length_bins'][k])

	if ('action_tokens' in kwargs) and (kwargs['action_tokens'] is not None): # actions are descriptions just like 'words'
		features_sparse += [features['action'][w] for w in kwargs['action_tokens'] if w in features['action'].keys()]
	if ('repetitions' in kwargs) and (kwargs['repetitions'] is not None): # not using words, only ratio+len
		(_, len_rep, ratio_rep) = kwargs['repetitions']
		for k in features['rep_length_bins'].keys():
			if len_rep <= float(k.split('-')[1]) and len_rep >= float(k.split('-')[0]):
				features_sparse.append(features['rep_length_bins'][k])
		for k in features['rep_ratio_bins'].keys():
			if len_rep <= float(k.split('-')[1]) and len_rep >= float(k.split('-')[0]):
				features_sparse.append(features['rep_ratio_bins'][k])
	
	# transforming features
	features_full = [1 if i in features_sparse else 0 for i in range(nb_features)]

	return features_full

def generate_features(data:pd.DataFrame, tag:str, 
						nb_occ:int, use_action:bool, use_repetitions:bool,
						bin_cut:int = 10) -> dict:
	"""Analyse data according to arguments passed and generate features_idx dictionary. Printing log data to console.
	"""
	# Also counting tags for knowledge's sake
	print("\nTag counts: ")
	count_tags = data[tag].value_counts().to_dict()
	for k in sorted(count_tags.keys()):
		print("{}: {}".format(k,count_tags[k]), end=" ; ")

	# Features: vocabulary (spoken)
	count_vocabulary = [y for x in data.tokens.tolist() for y in x] # flatten
	count_vocabulary = dict(Counter(count_vocabulary))
	# filtering features
	count_vocabulary = {k:v for k,v in count_vocabulary.items() if v > nb_occ}
	# turning vocabulary into numbered features - ordered vocabulary
	features_idx = {'words': {k:i for i, k in enumerate(sorted(count_vocabulary.keys()))}}
	print("\nThere are {} words in the features".format(len(features_idx['words'])))

	# Features: Speakers (+ logging counts)
	count_spk = dict(Counter(data['speaker'].tolist()))
	print("\nSpeaker counts: ")
	for k in sorted(count_spk.keys()):
		print("{}: {}".format(k,count_spk[k]), end=" ; ")
	features_idx['speaker'] = {k:(len(features_idx['words'])+i) for i, k in enumerate(sorted(count_spk.keys()))}
	
	# Features: sentence length (+ logging counts)
	data['len_bin'], bins = pd.qcut(data.turn_length, q=bin_cut, duplicates='drop', labels=False, retbins=True)
	print("\nTurn length splits: ")
	for i,k in enumerate(bins[:-1]):
		print("\tlabel {}: turns of length {}-{}".format(i,k, bins[i+1]))

	nb_feat = max([max(v.values()) for v in features_idx.values()])
	features_idx['length_bins'] = {"{}-{}".format(k, bins[i+1]):(nb_feat+i) for i, k in enumerate(bins[:-1])}
	features_idx['length'] = {i:(nb_feat+i) for i, _ in enumerate(bins[:-1]) }
	# parameters: duplicates: 'raise' raises error if bins are identical, 'drop' just ignores them (leading to the creation of larger bins by fusing those with identical cuts)
	# retbins = return bins (for debug) ; labels=False: only yield the position in the binning, not the name (simpler to create features)

	# Features: actions
	if use_action:
		count_actions = [y for x in data.action_tokens.tolist() for y in x] # flatten
		count_actions = dict(Counter(count_actions))
		# filtering features
		count_actions = {k:v for k,v in count_actions.items() if v > nb_occ}
		# turning vocabulary into numbered features - ordered vocabulary
		nb_feat = max([max(v.values()) for v in features_idx.values()])
		features_idx['action'] = {k:i+nb_feat for i, k in enumerate(sorted(count_actions.keys()))}
		print("\nThere are {} words in the actions".format(len(features_idx['action'])))	

	# Features: repetitions (reusing word from speech)
	if use_repetitions:
		nb_feat = max([max(v.values()) for v in features_idx.values()])
		# features esp for length & ratio - repeated words can use previously defined features
		# lengths
		_, bins = pd.qcut(data.nb_repwords, q=bin_cut, duplicates='drop', labels=False, retbins=True)
		features_idx['rep_length_bins'] = {"{}-{}".format(k, bins[i+1]):(nb_feat+i) for i, k in enumerate(bins[:-1])}
		# ratios
		_, bins = pd.qcut(data.ratio_repwords, q=bin_cut, duplicates='drop', labels=False, retbins=True)
		features_idx['rep_ratio_bins'] = {"{}-{}".format(k, bins[i+1]):(nb_feat+i) for i, k in enumerate(bins[:-1])}
		print("\nRepetition ratio splits: ")
		for i,k in enumerate(bins[:-1]):
			print("\tlabel {}: turns of length {}-{}".format(i,k, bins[i+1]))
	
	return features_idx

### BASELINE
def baseline_model(name:str, weights:dict, balance:bool):
	"""Create and update (if need be) model with weights
	"""
	models = {
		'SVC': svm.SVC(),
		'LSVC': svm.LinearSVC(), 
		'NB': naive_bayes.GaussianNB(), 
		'RF': ensemble.RandomForestClassifier(n_estimators=100)
	}
	if balance:
		try:
			models[name].set_params(class_weight=weights)
		except ValueError as e:
			if "Invalid parameter class_weight for estimator" in str(e): # GaussianNB has no such parameter for instance
				pass
			else: 
				raise e
		except Exception as e:
			raise e
	return models[name]


### REPORT
def plot_training(trainer, file_name):
	logs = pd.DataFrame(trainer.logparser.iterations) # initially list of dicts
	# columns: {'loss', 'error_norm', 'linesearch_trials', 'active_features', 'num', 'time', 'scores', 'linesearch_step', 'feature_norm'}
	# FYI scores is empty
	logs.set_index('num', inplace=True)
	for col in ['loss', 'active_features']:
		plt.figure()
		plt.plot(logs[col])
		plt.savefig(file_name+'/'+col+'.png')


#### MAIN
if __name__ == '__main__':
	args = argparser()
	print(args)

	# Definitions
	number_words_for_feature = args.nb_occurrences # default 5
	number_segments_length_feature = 10
	#number_segments_turn_position = 10 # not used for now
	training_tag = 'spa_'+args.keep_tag

	print("### Loading data:".upper())
	if args.format == 'txt':
		if args.txt_columns == []:
			raise TypeError('--txt_columns [col0] [col1] ... is required with format txt')
		args.use_action = args.use_action & ('action' in args.txt_columns)
		args.use_past_actions = args.use_past_actions & args.use_action
		data_train = openData(args.train, cut=args.cut, column_names=args.txt_columns, match_age=args.match_age, use_action = args.use_action, check_repetition=args.use_repetitions, use_past=args.use_past, use_pastact=args.use_past_actions)

	elif args.format == 'tsv':
		data_train = pd.read_csv(args.train, sep='\t', keep_default_na=False).reset_index(drop=False)
		args.use_action = args.use_action & ('action' in data_train.columns.str.lower())
		args.use_past_actions = args.use_past_actions & args.use_action
		data_train.rename(columns={col:col.lower() for col in data_train.columns}, inplace=True)
		data_train = data_add_features(data_train, use_action=args.use_action, match_age=args.match_age, check_repetition=args.use_repetitions, use_past=args.use_past, use_pastact=args.use_past_actions)
		training_tag = [x for x in data_train.columns if 'spa_' in x][0]
		args.training_tag = training_tag
	
	print("### Creating features:".upper())
	features_idx = generate_features(data_train, training_tag, args.nb_occurrences, args.use_action, args.use_repetitions, bin_cut=number_segments_length_feature)

	# creating crf features set for train
	data_train['features'] = data_train.apply(lambda x: word_to_feature(features_idx, x.tokens, x['speaker'], x.turn_length, 
												action_tokens=None if not args.use_action else x.action_tokens, 
												repetitions=None if not args.use_repetitions else (x.repeated_words, x.nb_repwords, x.ratio_repwords),
												past_tokens=None if not args.use_past else x.past,
												pastact_tokens=None if not args.use_past_actions else x.past_act), axis=1)

	if args.train_percentage < 1:
		# only take x % of the files
		train_files = data_train['file_id'].unique().tolist()
		train_subset = np.random.choice(len(train_files), size=int(len(train_files)*args.train_percentage), replace=False)
		train_files = [train_files[x] for x in train_subset]
		data_train = data_train[data_train['file_id'].isin(train_files)]

	# Once the features are done, groupby name and extract a list of lists
	# some "None" appear bc some illocutionary codes missing - however creating separations between data...
	grouped_train = data_train.dropna(subset=[training_tag]).groupby(by=['file_id']).agg({
		'features' : lambda x: [y for y in x],
		training_tag : lambda x: [y for y in x], 
		'index': min
	}) # listed by apparition order
	grouped_train = sklearn.utils.shuffle(grouped_train)

	# After that, train ---
	print("\n### Training starts.".upper())
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

	# Location for weight save
	name = os.path.join(os.getcwd(),('' if args.out is None else args.out), 
				'_'.join([ x for x in [training_tag, datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')] if x ])) # creating name with arguments, removing Nones in list
	print("Saving model at: {}".format(name))
	os.mkdir(name)
	trainer.train(os.path.join(name, 'model.pycrfsuite'))
	# plotting training curves
	plot_training(trainer, name)
	# dumping features
	with open(os.path.join(name, 'features.json'), 'w') as json_file:
		json.dump(features_idx, json_file)
	# dumping metadata
	with open(os.path.join(name, 'metadata.txt'), 'w') as meta_file:
		for arg in vars(args):
			meta_file.write("{0}:\t{1}\n".format(arg, getattr(args, arg)))
	
	# Baseline
	if args.baseline is not None:
		print("\nTraining and saving baseline model for comparison.")
		X = data_train.dropna(subset=[training_tag]).apply(lambda x: word_bs_feature(features_idx, x.tokens, x['speaker'], 
															x.turn_length, 
															action_tokens=None if not args.use_action else x.action_tokens, 
															repetitions=None if not args.use_repetitions else (x.repeated_words, x.nb_repwords, x.ratio_repwords)
															), axis=1)
		y = data_train.dropna(subset=[training_tag])[training_tag].tolist()
		weights = dict(Counter(y))
		# ID from label - bidict
		labels = dataset_labels(training_tag.upper(), add_empty_labels=True) # TODO: remove line with unreadable labels
		# transforming
		X = np.array(X.tolist())
		y = np.array([labels[lab] for lab in y]) # to ID
		weights = {labels[lab]:v/len(y) for lab, v in weights.items()} # update weights as proportion, ID as labels
		mdl = baseline_model(args.baseline, weights, args.balance_ex) # Taking imbalance into account
		mdl.fit(X,y)
		dump(mdl, os.path.join(name, 'baseline.joblib'))
		print("Done.")