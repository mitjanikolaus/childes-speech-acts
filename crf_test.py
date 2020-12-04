import os
import pickle
import argparse
from collections import Counter
import json
import ast
from typing import Union, Tuple
from bidict import bidict

import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
import pycrfsuite

from preprocess import SPEECH_ACT
from crf_train import add_feature_columns, get_features_from_row, crf_predict


def argparser():
	argparser = argparse.ArgumentParser(description='Test a previously trained CRF')
	argparser.add_argument('data', type=str, help="file listing all dialogs")
	argparser.add_argument(
		"--test-ratio",
		type=float,
		default=.2,
		help="Ratio of given dataset to be used to testing",
	)
	argparser.add_argument('--txt_columns', nargs='+', type=str, default=[], help=""".txt columns name (in order); most basic txt is ['spa_all', 'ut', 'time', 'speaker', 'sentence']""")
	argparser.add_argument('--match_age', type=int, nargs='+', default=None, help="ages to match data to - for split analysis")
	argparser.add_argument('--model', '-m', required=True, type=str, default=None, help="folder containing model, features and metadata")
	# parameters for training/testing:
	argparser.add_argument('--col_ages', type=str, default=None, help="if not None, plot evolution of accuracy over age groups")
	argparser.add_argument('--consistency_check', action="store_true", help="whether 'child' column matters in testing data.")
	argparser.add_argument('--prediction_mode', choices=["raw", "exclude_ool"], default="exclude_ool", type=str, help="Whether to predict with NOL/NAT/NEE labels or not.")

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
			except Exception as e:
				raise e

	return args

def baseline_predict(model, data, labels:bidict, mode:str='raw',
				exclude_labels:list = ['NOL', 'NAT', 'NEE']) -> Union[list, Tuple[list, pd.DataFrame]]:
	"""Return predictions for the test data. 3 modes for return:
		* Return raw predictions (raw)
		* Return predictions with only valid tags (exclude_ool)
		* Return predictions (valid tags) and probabilities for each class (rt_proba)
	"""
	if mode not in ['raw', 'exclude_ool', 'rt_proba']:
		raise ValueError(f"mode must be one of raw|exclude_ool|rt_proba; currently {mode}")
	
	if mode == 'raw':
		# still need to transform class to label
		return [labels.inverse[x] for x in model.predict(data)] # predicting int version of labels, not ordered number

	y_proba = model.predict_proba(data) # predict proba to remove extra labels
	classes_names = [labels.inverse[x] for x in model.classes_] # proba ordered by classes_ => label
	y_proba = pd.DataFrame(y_proba, columns=classes_names)
	# filter & predict
	y_pred = y_proba[[col for col in y_proba.columns if col not in exclude_labels]].idxmax(axis=1).tolist()

	if mode == 'rt_proba':
		return y_pred, y_proba
	return y_pred # else

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


	# Loading model
	name = args.model
	if os.path.isdir(name):
		linker = '/'
		if name[-1] == '/':
			name = name[:-1]
	elif os.path.isfile(name + '_model.pycrfsuite'):
		linker = '_'
		# get args from name if possible
		try:
			rp, args.use_action, args.use_repetitions, args.use_past, args.use_past_actions = name.split('/')[-1].split('_')
		except ValueError as e:
			if 'unpack' in str(e):
				raise AttributeError("Cannot find model metadata - args need to be set.")
		args.baseline = None # default
	else:
		raise FileNotFoundError(f"Cannot find model {name}.")
	# update paths for input/output
	features_path = name + linker + 'features.json'
	model_path = name + linker + 'model.pycrfsuite'
	report_path = name + linker + args.data.replace('/', '_')+'_report.xlsx'
	plot_path = name + linker + args.data.split('/')[-1]+'_agesevol.png'

	# Loading data
	data = pd.read_pickle(args.data)

	data = add_feature_columns(
		data,
		use_action=args.use_action,
		match_age=args.match_age,
		check_repetition=args.use_repetitions,
		use_past=args.use_past,
		use_pastact=args.use_past_actions,
	)

	_, data_test = train_test_split(data, test_size=args.test_ratio, shuffle=False)

	# Loading features
	with open(features_path, 'r') as json_file:
		features_idx = json.load(json_file)
	data_test['features'] = data_test.apply(lambda x: get_features_from_row(features_idx, x.tokens, x['speaker'], x.turn_length,
																			action_tokens=None if not args.use_action else x.action_tokens,
																			repetitions=None if not args.use_repetitions else (x.repeated_words, x.nb_repwords, x.ratio_repwords),
																			past_tokens=None if not args.use_past else x.past,
																			pastact_tokens=None if not args.use_past_actions else x.past_act), axis=1)

	# Predictions
	tagger = pycrfsuite.Tagger()
	tagger.open(model_path)
	
	data_test.dropna(subset=[SPEECH_ACT], inplace=True)
	X_dev = data_test.groupby(by=['file_id']).agg({ 
		'features' : lambda x: [y for y in x],
		'index': min
	})
	# parameter 'raw' vs 'exclude_ool': remove ['NOL', 'NAT', 'NEE'] from predictions, predict closest label
	y_pred = crf_predict(tagger, X_dev.sort_values('index', ascending=True)['features'], mode=args.prediction_mode) 
	data_test['y_pred'] = [y for x in y_pred for y in x] # flatten
	data_test['y_true'] = data_test[SPEECH_ACT]
	data_test['pred_OK'] = data_test.apply(lambda x: (x.y_pred == x.y_true), axis=1)
	# only report on tags where y_true != NOL, NAT, NEE
	data_crf = data_test[~data_test['y_true'].isin(['NOL', 'NAT', 'NEE'])]
	# reports
	report, mat, acc, cks = bio_classification_report(data_crf['y_true'].tolist(), data_crf['y_pred'].tolist())
	states, transitions = features_report(tagger)
	
	int_cols = ['file_id', 'speaker'] + ([args.col_ages] if args.col_ages is not None else []) + [x for x in data_test.columns if 'spa_' in x] + (['child'] if args.consistency_check else [])+ ['y_true', 'y_pred', 'pred_OK']

	report_d = {
		'test_data': data_crf[int_cols],
		'classification_report': report.T,
		'confusion_matrix': mat,
		'weights': states, 
		'learned_transitions': transitions.pivot(index='label_from', columns='label', values='likelihood') 
	}

	pickle.dump(report.T, open("data/classification_scores_crf.p","wb"))

	if args.col_ages is not None:
		plot_testing(data_test, plot_path, args.col_ages)

	# Write excel with all reports
	report_to_file(report_d, report_path)