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
	$ python crf_train.py --train ttv/list_train_conv --test ttv/list_test_conv --dev ttv/list_dev_conv --format txt --keep_tag 2
	$ python crf_train.py --train ttv/list_train_conv --test ttv/list_test_conv --dev ttv/list_valid_conv --format txt --keep_tag 2 --use_action --txt_columns spa_all utterance time_stamp speaker sentence action --use_repetitions
"""
import os
import sys
import random
import codecs
import argparse
import time, datetime
from collections import Counter
import json

import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite

### Tag functions
from utils import *


#### Read Data functions
def openData(list_file:str, cut=100000, column_names=['all', 'ut', 'time', 'speaker', 'sentence'], match_age=None, use_action=False, check_repetition=False):
	"""
	Input:
	------
	list_file: `str`
		location of file containing train/dev/test txt files

	cut: `int`
		number of files to keep
	
	column_names: `list`
		list of features in the text file
	
	match_age: `list`
		list of ages to match column age_months to - if needed by later analysis. Matching column to closest value in list.
	
	use_action: `bool`
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
		tmp = pd.read_csv(file_name, sep="\t", names=column_names)
		# either removing empty sentences or replacing with ""
		tmp = tmp[~pd.isna(tmp.sentence)]
		#tmp['sentence'] = tmp.sentence.fillna("")
		tmp['file_id'] = file_name
		tmp['index'] = i
		p.append(tmp)
	p = pd.concat(p)
	# Changing locutors: INV/FAT become mother
	p['speaker'] = p['speaker'].apply(lambda x: x if x in ['CHI', 'MOT'] else 'MOT')
	# Adding features
	p = data_add_features(p, use_action=use_action, match_age=match_age, check_repetition=check_repetition)
	# Splitting tags
	for col_name, t in zip(['spa_1', 'spa_2', 'spa_2a'], ['first', 'second', 'adapt_second']):
		p[col_name] = p['spa_all'].apply(lambda x: select_tag(x, keep_part=t)) # creating columns with different tags
	# Return
	return p


#### Features functions
def data_add_features(p:pd.DataFrame, use_action=False, match_age=None, check_repetition=False):
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
	p['tokens'] = p.sentence.apply(lambda x: x.lower().split())
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
	if check_repetition:
		p['prev_file'] = p.file_id.shift(1).fillna(p.file_id.iloc[0])
		p['prev_spk'] = p.speaker.shift(1).fillna(p.speaker.iloc[0])
		p['prev_st'] = p.tokens.shift(1)#.fillna(p.tokens.iloc[0]) # doesn't work - fillna doesn't accept a list as value
		p['prev_st'].iloc[0] = p.tokens.iloc[0]
		p['repeated_words'] = p.apply(lambda x: [w for w in x.tokens if w in x.prev_st] if (x.prev_spk != x.speaker) and (x.file_id == x.prev_file) else [], axis=1)
		p['nb_repwords'] = p.repeated_words.apply(len)
		p['ratio_repwords'] = p.nb_repwords/p.turn_length
		p = p[[col for col in p.columns if col not in ['prev_spk', 'prev_st', 'prev_file']]]
	# return Dataframe
	return p


def word_to_feature(features:dict, spoken_tokens:list, speaker:str, ln:int, action_tokens=None, repetitions=None):
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
	
	action_tokens: `list`
		data action, default None if actions are not taken into account
	
	Output:
	-------
	feat_glob: `dict`
		dictionary of same shape as feature, but only containing features relevant to data line
	"""
	#feat_glob = {'words': Counter([w for w in spoken_tokens if (w in features['words'].keys())]) if (features['words'] is not None) else Counter(spoken_tokens)} # Can unknown words be included (test/dev) in this?
	feat_glob = { 'words': Counter([w for w in spoken_tokens if (w in features['words'].keys())]) }
	feat_glob['speaker'] = {speaker:1.0}
	feat_glob['length'] = {k:(1 if ln <= float(k.split('-')[1]) and ln >= float(k.split('-')[0]) else 0) for k in features['length_bins'].keys()}
	#feat_glob['length<=i'] = {k:(1 if ln <= float(k.split('-')[1]) else 0) for k in feat_subset.keys()}
	#feat_glob['length>=i'] = {k:(1 if ln >= float(k.split('-')[0]) else 0) for k in feat_subset.keys()}
	if action_tokens is not None:
		# actions are descriptions just like 'words'
		feat_glob['actions'] = Counter([w for w in action_tokens if (w in features['action'].keys())]) #if (features['action'] is not None) else Counter(action_tokens)
	if repetitions is not None:
		(rep_words, len_rep, ratio_rep) = repetitions
		feat_glob['repeated_words'] = Counter([w for w in rep_words if (w in features['words'].keys())])
		feat_glob['rep_length'] = {k:(1 if len_rep <= float(k.split('-')[1]) and len_rep >= float(k.split('-')[0]) else 0) for k in features['rep_length_bins'].keys()}
		feat_glob['rep_ratio'] = {k:(1 if ratio_rep <= float(k.split('-')[1]) and ratio_rep >= float(k.split('-')[0]) else 0) for k in features['rep_ratio_bins'].keys()}

	return feat_glob
#	# words
#	for w in l:
#		if w in features.keys():
#			s.add(features[w])
#	# locutor
#	s.add(features[locutor])
#	# sentence length bin
#	feat_subset = {k: v for k, v in features.items() if isinstance(k, str) and (re.search(re.compile('[0-9]{1,2}_[0-9]{1,2}'), k) is not None)}
#	for i, (k,v) in enumerate(feat_subset.items()):
#		min_len, max_len = k.split('_')
#		if i == 0 and ln <= int(min_len):
#			s.add(v)
#		elif ln <= int(max_len) and ln > int(min_len):
#			s.add(v)
#		elif i == len(feat_subset) and ln > int(max_len):
#			s.add(v)
#	return s
	


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

	print("==> Accuracy: {0:.3f}".format(acc))
	# using both as index in case not the same labels in it
	return pd.DataFrame(cr), pd.DataFrame(cm, index=sorted(set(y_true+y_pred)), columns=sorted(set(y_true+y_pred))), acc

def report_to_file(dfs:dict, file_location:str):
	"""Looping on each pd.DataFrame to log to excel
	"""
	writer = pd.ExcelWriter(file_location)
	for name, data in dfs.items():
		data.to_excel(writer, sheet_name=name)
	writer.save()

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
	argparser = argparse.ArgumentParser(description='Train a CRF and test it.', formatter_class=argparse.RawTextHelpFormatter)
	# Data files
	argparser.add_argument('--test', type=str, required=True, help="file listing test dialogs")
	argparser.add_argument('--train', type=str, required=True, help="file listing train dialogs")
	argparser.add_argument('--dev', type=str, required=True, help="file listing dev dialogs")
	argparser.add_argument('--format', '-f', choices=['txt', 'tsv'], required=True, help="data file format - adapt reading")
	argparser.add_argument('--txt_columns', nargs='+', type=str, default=[], help=""".txt columns name (in order); most basic txt is ['spa_all', 'ut', 'time', 'speaker', 'sentence']""")
	argparser.add_argument('--match_age', type=int, nargs='+', default=None, help="ages to match data to - for split analysis")
	# Operations on data
	argparser.add_argument('--keep_tag', choices=['all', '1', '2', '2a'], default="all", help="keep first part / second part / all tag")
	argparser.add_argument('--cut', type=int, default=1000000, help="if specified, use the first n train dialogs instead of all.")
	argparser.add_argument('--split_ages', type=bool, default=False, help="if True, training separately the different group ages")
	argparser.add_argument('--split_loc', type=bool, default=False, help="if True, training separately adult and child dialogs")
	argparser.add_argument('--out', type=str, default='results', help="where to write .crfsuite model file")
	argparser.add_argument('--error', action='store_true')
	# parameters for training/testing:
	argparser.add_argument('--nb_occurrences', '-noc', type=int, default=5, help="number of minimum occurrences for word to appear in features")
	argparser.add_argument('--use_action', '-act', action='store_true', help="whether to use action features to train the algorithm, if they are in the data")
	argparser.add_argument('--use_repetitions', '-rep', action='store_true', help="whether to check in data if words were repeated from previous sentence, to train the algorithm")
	argparser.add_argument('--weights_loc', '-w', type=str, default=None, help="pattern name for the previous learning parameters")

	args = argparser.parse_args()
	print(args)
	print("Training starts.")

	# Definitions
	number_words_for_feature = args.nb_occurrences # default 5
	number_segments_length_feature = 10
	#number_segments_turn_position = 10 # not used for now

	if args.format == 'txt':
		if args.txt_columns == []:
			raise TypeError('--txt_columns [col0] [col1] ... is required with format txt')
		use_action = args.use_action & ('action' in args.txt_columns)
		data_train = openData(args.train, cut=args.cut, column_names=args.txt_columns, match_age=args.match_age, use_action = use_action, check_repetition=args.use_repetitions)
		data_test = openData(args.test, column_names=args.txt_columns, match_age=args.match_age, use_action = use_action, check_repetition=args.use_repetitions)
		data_dev = openData(args.dev, column_names=args.txt_columns, match_age=args.match_age, use_action = use_action, check_repetition=args.use_repetitions)
	elif args.format == 'tsv':
		data_train = pd.read_csv(args.train, sep='\t')
		data_test = pd.read_csv(args.test, sep='\t')
		data_dev = pd.read_csv(args.dev, sep='\t')
		use_action = args.use_action & ('action' in data.columns)
		for data in [data_train, data_dev, data_test]:
			data.rename(columns={col:col.lower() for col in data.columns})
			data = data_add_features(data, action=use_action, match_age=args.match_age, check_repetition=args.use_repetitions)
	
	training_tag = 'spa_'+args.keep_tag
	count_tags = data_train[training_tag].value_counts().to_dict()
	# printing log data:
	print("\nTag counts: ")
	for k in sorted(count_tags.keys()):
		print("{}: {}".format(k,count_tags[k]), end=" ; ")

	count_vocabulary = [y for x in data_train.tokens.tolist() for y in x] # flatten
	count_vocabulary = dict(Counter(count_vocabulary))
	# filtering features
	count_vocabulary = {k:v for k,v in count_vocabulary.items() if v > args.nb_occurrences}
	# turning vocabulary into numbered features - ordered vocabulary
	features_idx = {'words': {k:i for i, k in enumerate(sorted(count_vocabulary.keys()))}}
	print("\nThere are {} words in the features".format(len(features_idx['words'])))

	# adding other features:
	count_spk = dict(Counter(data_train['speaker'].tolist()))
	# printing log data:
	print("\nSpeaker counts: ")
	for k in sorted(count_spk.keys()):
		print("{}: {}".format(k,count_spk[k]), end=" ; ")
	#features_idx = {**features_idx, **{k:(len(features_idx)+i) for i, k in enumerate(sorted(count_spk.keys()))}}
	features_idx['speaker'] = {k:(len(features_idx['words'])+i) for i, k in enumerate(sorted(count_spk.keys()))}
	
	#max_turn_length = data_train.turn_length.max()
	#count_turn_length = dict(Counter(data_train.turn_length.tolist()))
	data_train['len_bin'], bins = pd.qcut(data_train.turn_length, q=number_segments_length_feature, duplicates='drop', labels=False, retbins=True)
	# printing log data:
	print("\nTurn length splits: ")
	for i,k in enumerate(bins[:-1]):
		print("\tlabel {}: turns of length {}-{}".format(i,k, bins[i+1]))
	#features_idx = {
	#	**features_idx, 
	#	**{"{}-{}".format(k, bins[i+1]):(len(features_idx)+i) for i, k in enumerate(bins[:-1])}, # bin labels (FYI)
	#	**{i:(len(features_idx)+i) for i, _ in enumerate(bins[:-1]) } # actual labels (in data)
	#}
	nb_feat = max([max(v.values()) for v in features_idx.values()])
	features_idx['length_bins'] = {"{}-{}".format(k, bins[i+1]):(nb_feat+i) for i, k in enumerate(bins[:-1])}
	features_idx['length'] = {i:(nb_feat+i) for i, _ in enumerate(bins[:-1]) }
	# parameters: duplicates: 'raise' raises error if bins are identical, 'drop' just ignores them (leading to the creation of larger bins by fusing those with identical cuts)
	# retbins = return bins (for debug) ; labels=False: only yield the position in the binning, not the name (simpler to create features)

	# features: actions
	if use_action:
		count_actions = [y for x in data_train.action_tokens.tolist() for y in x] # flatten
		count_actions = dict(Counter(count_actions))
		# filtering features
		count_actions = {k:v for k,v in count_actions.items() if v > args.nb_occurrences}
		# turning vocabulary into numbered features - ordered vocabulary
		#action_features = {k:i+len(features_idx) for i, k in enumerate(sorted(count_actions.keys()))}
		#features_idx = {
		#	**features_idx, 
		#	**action_features
		#} 
		nb_feat = max([max(v.values()) for v in features_idx.values()])
		features_idx['action'] = {k:i+nb_feat for i, k in enumerate(sorted(count_actions.keys()))}
		print("\nThere are {} words in the actions".format(len(features_idx['action'])))	

	if args.use_repetitions:
		nb_feat = max([max(v.values()) for v in features_idx.values()])
		# features esp for length & ratio - repeated words can use previously defined features
		# lengths
		_, bins = pd.qcut(data_train.nb_repwords, q=number_segments_length_feature, duplicates='drop', labels=False, retbins=True)
		features_idx['rep_length_bins'] = {"{}-{}".format(k, bins[i+1]):(nb_feat+i) for i, k in enumerate(bins[:-1])}
		# ratios
		_, bins = pd.qcut(data_train.ratio_repwords, q=number_segments_length_feature, duplicates='drop', labels=False, retbins=True)
		features_idx['rep_ratio_bins'] = {"{}-{}".format(k, bins[i+1]):(nb_feat+i) for i, k in enumerate(bins[:-1])}
		print("\nRepetition ratio splits: ")
		for i,k in enumerate(bins[:-1]):
			print("\tlabel {}: turns of length {}-{}".format(i,k, bins[i+1]))

	# creating features set for train, dev, test
	for data in [data_train, data_dev, data_test]:
		data['features'] = data.apply(lambda x: word_to_feature(features_idx, x.tokens, x['speaker'], x.turn_length, None if not use_action else x.action_tokens, None if not args.use_repetitions else (x.repeated_words, x.nb_repwords, x.ratio_repwords)), axis=1)

	# Once the features are done, groupby name and extract a list of lists
	# some "None" appear bc some illocutionary codes missing - however creating separations between data...
	grouped_train = data_train.dropna(subset=[training_tag]).groupby(by=['file_id']).agg({
		'features' : lambda x: [y for y in x],
		training_tag : lambda x: [y for y in x], 
		'index': min
	}) # listed by apparition order
	grouped_train = sklearn.utils.shuffle(grouped_train)
	grouped_train.to_excel('test.xlsx')

	if args.weights_loc is None:
		# After that, train ---
		trainer = pycrfsuite.Trainer(verbose=True)
		# Adding data
		for idx, file_data in grouped_train.iterrows():
			try:
				trainer.append(file_data['features'], file_data[training_tag]) # X_train, y_train
			except:
				data_train[data_train.file == idx].to_excel('error_{}.xlsx'.format(idx.replace('/', '_')))
				print(idx)
		# Parameters
		trainer.set_params({
				'c1': 1,   # coefficient for L1 penalty
				'c2': 1e-3,  # coefficient for L2 penalty
				'max_iterations': 50,  # stop earlier
				'feature.possible_transitions': True # include transitions that are possible, but not observed
		})
		# Location for weight save
		name = os.path.join(os.getcwd(),('' if args.out is None else args.out), 
				'_'.join([ x for x in [training_tag, 'sa' if args.split_ages else None, 'sl' if args.split_loc else None, datetime.datetime.now().strftime('%Y-%m-%d-%H%M')] if x ])) # creating name with arguments, removing Nones in list
		os.mkdir(name)
		trainer.train(os.path.join(name, 'model.pycrfsuite'))
		# plotting training curves
		plot_training(trainer, name)
		# dumpin features
		with open(os.path.join(name, 'features.json'), 'w') as json_file:
			json.dump(features_idx, json_file)
	else:
		name = args.weights_loc
		# loading features
		with open(os.path.join(name, 'features.json'), 'r') as json_file:
			features_idx = json.load(json_file)
		for data in [data_dev, data_test]:
			data['features'] = data.apply(lambda x: word_to_feature(features_idx, x.tokens, x['speaker'], x.turn_length, None if not use_action else x.action_tokens, None if not args.use_repetitions else (x.repeated_words, x.nb_repwords, x.ratio_repwords)), axis=1)

	# Predictions
	tagger = pycrfsuite.Tagger()
	tagger.open(os.path.join(name,'model.pycrfsuite'))
	
	# creating data
	data_dev.dropna(subset=[training_tag], inplace=True)
	X_dev = data_dev.groupby(by=['file_id']).agg({ 
		'features' : lambda x: [y for y in x],
		'index': min
	})
	y_pred = [tagger.tag(xseq) for xseq in X_dev.sort_values('index', ascending=True)['features']]
	data_dev['y_pred'] = [y for x in y_pred for y in x] # flatten
	data_dev['y_true'] = data_dev[training_tag]
	data_dev['pred_OK'] = data_dev.apply(lambda x: (x.y_pred == x.y_true), axis=1)
	# reports
	report, mat, acc = bio_classification_report(data_dev['y_true'].tolist(), data_dev['y_pred'].tolist())
	states, transitions = features_report(tagger)
	
	report_to_file({
		'test_data': data_dev[['file_id', 'speaker', 'spa_1', 'spa_2', 'spa_2a', 'y_true', 'y_pred', 'pred_OK']],
		'classification_report': report.T,
		'confusion_matrix': mat,
		'weights': states, 
		'learned_transitions': transitions.pivot(index='label_from', columns='label', values='likelihood') 
	}, name+'/report.xlsx')