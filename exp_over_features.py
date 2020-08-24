#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comparing training depending on features

Execute:
    $ python exp_over_features.py ttv/childes_ne_train_spa_2.tsv ttv/childes_ne_test_spa_2.tsv -f tsv
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
from utils import dataset_labels, ILLOC
from crf_train import openData, data_add_features, word_to_feature, word_bs_feature, generate_features
from crf_test import bio_classification_report, report_to_file

#### Read Data functions
def argparser():
    """Creating arparse.ArgumentParser and returning arguments
    """
    argparser = argparse.ArgumentParser(description='Train a CRF and test it.', formatter_class=argparse.RawTextHelpFormatter)
    # Data files
    argparser.add_argument('train', type=str, help="file listing train dialogs")
    argparser.add_argument('test', type=str, help="file listing train dialogs")
    argparser.add_argument('--format', '-f', choices=['txt', 'tsv'], required=True, help="data file format - adapt reading")
    argparser.add_argument('--txt_columns', nargs='+', type=str, default=[], help=""".txt columns name (in order); most basic txt is ['spa_all', 'ut', 'time', 'speaker', 'sentence']""")
    # Operations on data
    argparser.add_argument('--match_age', type=int, nargs='+', default=None, help="ages to match data to - for split analysis")
    argparser.add_argument('--keep_tag', choices=['all', '1', '2', '2a'], default="all", help="keep first part / second part / all tag")
    argparser.add_argument('--out', type=str, default='results', help="where to write .crfsuite model file")
    # parameters for training:
    argparser.add_argument('--nb_occurrences', '-noc', type=int, default=5, help="number of minimum occurrences for word to appear in features")
    argparser.add_argument('--verbose', action="store_true", help="Whether to display training iterations output.")

    args = argparser.parse_args()
    return args

#### Report
def plot_evolution(report: pd.DataFrame, location:str, reference:str, kind:str = 'f1-score', figsize:tuple = (20,5)):
    """Generates and saves two graphs: 
        * one with overall accuracy for each label for each experiment (between 0 and 1)
        * one with accuracy relative to the first experiment (between -1 and 1)
    
    Further reading: 
        * artists: https://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box
    """
    if kind not in ['precision', 'recall', 'f1-score']:
        raise ValueError("'kind' must be one of 'precision', 'recall', 'f1-score'")
    # select and rename columns
    cols = [col.replace(kind+'_', '') for col in report.columns if kind in col]
    report.rename(columns={col:col.replace(kind+'_', '') for col in report.columns if kind in col}, inplace=True)
    report.sort_values(by=reference, ascending=False, inplace=True)
    # drop non labels in rows
    try:
        report.drop(['accuracy', 'macro avg', 'weighted avg'], inplace=True)
    except:
        pass
    
    styles = {col:('.-' if col == reference else '.') for col in cols}
    # Figure 1
    fig, ax = plt.subplots(figsize=figsize)
    # xticks set manually - pandas bug
    report[cols].fillna(0).reset_index(drop=True).plot(ax = ax, style=styles)
    ax.set_xticks(range(len(report.index)))
    xtext = ax.set_xticklabels(report.index, rotation=270) 
    plt.savefig(os.path.join(location, 'evol'+'.png'), bbox_extra_artists=(xtext), bbox_inches='tight')
    
    # Figure 2
    fig, ax = plt.subplots(figsize=figsize)
    ref = report[reference].tolist()
    for col in cols:
        report[col] = report[col] - ref
    report[cols].fillna(0).reset_index(drop=True).plot(ax = ax, style=styles)
    ax.set_xticks(range(len(report.index)))
    xtext = ax.set_xticklabels(report.index, rotation=270)
    plt.savefig(os.path.join(location, 'evol_comparison'+'.png'), bbox_extra_artists=(xtext), bbox_inches='tight')


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
        elif 'action' not in args.txt_columns:
            raise ValueError("in order to test the impact of actions, they must be in the data")
        # Loading with actions and repetitions in order to use them later
        data_train = openData(args.train, cut=args.cut, column_names=args.txt_columns, match_age=args.match_age, use_action = True, check_repetition=True)
        data_test = openData(args.test, cut=args.cut, column_names=args.txt_columns, match_age=args.match_age, use_action = True, check_repetition=True)

    elif args.format == 'tsv':
        data_train = pd.read_csv(args.train, sep='\t').reset_index(drop=False)
        if 'action' not in data_train.columns.str.lower():
            raise ValueError("in order to test the impact of actions, they must be in the data")
        # Loading with actions and repetitions in order to use them later
        data_train.rename(columns={col:col.lower() for col in data_train.columns}, inplace=True)
        data_train = data_add_features(data_train, use_action=True, match_age=args.match_age, check_repetition=True)
        # Same for test
        data_test = pd.read_csv(args.test, sep='\t').reset_index(drop=False)
        data_test.rename(columns={col:col.lower() for col in data_test.columns}, inplace=True)
        data_test = data_add_features(data_test, use_action=True, match_age=args.match_age, check_repetition=True)
        # Parameters
        training_tag = [x for x in data_train.columns if 'spa_' in x][0]
        args.training_tag = training_tag
    

    logger = [] # Results
    freport = [] # Dataframes to save
    name = os.path.join(os.getcwd(),('' if args.out is None else args.out), 
                '_'.join([ x for x in [os.path.basename(__file__).replace('.py',''), training_tag, datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')] if x ])) # Location for weight save
    os.mkdir(name)
    # Training & Testing
    for use_action in [False, True]:
        for use_rep in [False, True]:
            pat = '_'.join(['act' if use_action else 'no-act', 'rep' if use_rep else 'no-rep'])
            nm = os.path.join(name, pat) 
            # generating features
            features_idx = generate_features(data_train, training_tag, args.nb_occurrences, use_action, use_rep)
            # creating crf features for train
            data_train['features'] = data_train.apply(lambda x: word_to_feature(features_idx, x.tokens, x['speaker'], x.turn_length, None if not use_action else x.action_tokens, None if not use_rep else (x.repeated_words, x.nb_repwords, x.ratio_repwords)), axis=1)

            # groupby to create training data
            grouped_train = data_train.dropna(subset=[training_tag]).groupby(by=['file_id']).agg({
                'features' : lambda x: [y for y in x],
                training_tag : lambda x: [y for y in x], 
                'index': min
            }) # listed by apparition order
            grouped_train = sklearn.utils.shuffle(grouped_train)

            ### Training
            print(f"\n### Training {pat}: start.".upper())
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
            print(f"Saving model at: {nm}")
    
            trainer.train(nm +'_model.pycrfsuite')
            with open(nm+'_features.json', 'w') as json_file: # dumping features
                json.dump(features_idx, json_file)
        
            ### Testing - looping over test samples
            tagger = pycrfsuite.Tagger()
            tagger.open(nm +'_model.pycrfsuite')
            # Features
            data_test['features'] = data_test.apply(lambda x: word_to_feature(features_idx, x.tokens, x['speaker'], x.turn_length, None if not use_action else x.action_tokens, None if not use_rep else (x.repeated_words, x.nb_repwords, x.ratio_repwords)), axis=1)

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

            logger.append({'mode':pat, 'acc':acc, 'kappa':cks})
            freport.append(report.T.rename(columns={col:(col+'_'+pat) for col in report.T.columns}))
    
    res_comp = pd.DataFrame(logger)
    rep_comp = pd.concat(freport, axis=1)
    rep_comp = pd.concat([rep_comp, ILLOC], axis=1)
    report_to_file({ 
        'comparison': res_comp.set_index('mode'),
        'precision_evolution': rep_comp[[col for col in rep_comp.columns if 'precision' in col]],
        'recall_evolution': rep_comp[[col for col in rep_comp.columns if 'recall' in col]],
        'f1_evolution': rep_comp[[col for col in rep_comp.columns if 'f1' in col]],
    }, os.path.join(name, 'report.xlsx'))

    # operations on rep_comp before plotting - adding detailed index
    rep_comp.drop(['accuracy', 'macro avg', 'weighted avg'], inplace=True)
    rep_comp.reset_index(inplace=True, drop=False)
    rep_comp['concat'] = rep_comp.apply(lambda x: x['index'] + ' - ' + x['Name'] + ' (' + x['Description'] + ')', axis=1)
    rep_comp.set_index('concat', inplace=True)
    # plot evolution
    plot_evolution(rep_comp, name, 'no-act_no-rep') 

    with open(os.path.join(name, 'metadata.txt'), 'w') as meta_file: # dumping metadata
        for arg in vars(args):
            meta_file.write("{0}:\t{1}\n".format(arg, getattr(args, arg)))
        meta_file.write("{0}:\t{1}\n".format("Experiment", "Features"))
