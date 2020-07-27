""" Formatting previously extracted data into datasets with correct format for training/testing

Examples:
    With training data (NewEngland corpus):
    >>> python extract_data.py formatted/NewEngland/data.csv --output_format one_tsv --ttv_filepattern childes_{}.tsv --select_data file_id spa_1 spa_2 spa_2a speaker sentence
    With test data (Bates corpus):
    >>> python extract_data.py formatted/Bates/data.csv --keep_empty_spa --output_format one_tsv --ttv_filepattern 'bates_{}.tsv'
"""
import json
import os, sys
from collections import Counter
import numpy as np
import pandas as pd
import time, datetime
import argparse
import re

from utils import *

def sep_loop(df, line_function, ttv_writer, remove_empty_tags=True):
    # extract file <-> ttv columns, set as index, series to dict
    sets = df[['file_id', 'ttv']].drop_duplicates().set_index('file_id').to_dict()['ttv']
    # either remove empty tags or replace them with placeholder
    ss = [col for col in line_function if 'spa_' in col]
    if remove_empty_tags:
        data.dropna(subset = ss, inplace=True)
    else:
        data[ss] = data[ss].fillna(value='None')
    for k,v in sets.items():
        ttv_writer[v].write(k.replace('.json', '.txt')+'\n')
        df[df['file_id'] == k][line_function].to_csv(k.replace('.json', '.txt'), sep='\t', header=False, index=False)

def one_loop(df, line_function, ttv_writer, remove_empty_tags=True):
    sub_data = {k: df[df['ttv'] == k] for k in range(0,3)}
    for k,v in ttv_writer.items():
        for columns, filepath in zip(line_function, v):
            tag = [col for col in columns if 'spa_' in col]
            if remove_empty_tags:
                sub_data[k].dropna(subset = tag, inplace=True)
            else:
                sub_data[k][tag] = sub_data[k][tag].fillna(value='None')
            sub_data[k][columns].rename(columns={col:col.upper() for col in columns}).to_csv(filepath, sep='\t', index=False)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Extract and transform data.', formatter_class=argparse.RawTextHelpFormatter)
    # Data files
    argparser.add_argument('csv_loc', type=str, help='location of the previously extracted data')
    argparser.add_argument('--output_format', choices=['one_tsv', 'sep_txt'], help="""how to output the data based on the algorithm that will be applied next: 
        * 'one_tsv': HDF5 preparation for RNN
        * 'sep_txt': TXT preparation for CRF
    """)
    # Extraction parameters
    argparser.add_argument('--keep_empty_text', action='store_true', help='whether to keep segments without any speech (might have acts) in final data')
    argparser.add_argument('--keep_empty_spa', action='store_true', help="whether to keep segments without spa tags (analyzing unlabeled corpuses)")
    argparser.add_argument('--select_data', choices=["utterance", "spa_all", "spa_1", "spa_2", "spa_2a", "time_stamp", "speaker", "sentence", "lemmas", "pos", "action", "file_id"], nargs='+', help="ordered features for output - will be adapted to generate 3 files for ")
    argparser.add_argument('--replace_names', action='store_true', help="whether to replace locutors callings by tag; applicable on columns 'sentence' and 'lemmas'")
    # Train/test/validation
    argparser.add_argument('--ttv_split', nargs=3, type=float, default=[0.6, 0.3, 0.1], help="percentage of files going to train/test/validation sets; if --keep_empty_spa is False, test is all of the data")
    argparser.add_argument('--ttv_filepattern', type=str, default="list_{}_conv", help="file pattern (filled with train/test/valid + other patterns if need be) for ttv files - default 'list_{}_conv'.format(XXX)")

    args = argparser.parse_args()
    if args.select_data is None:
        if args.output_format == 'sep_txt':
            args.select_data = ["spa_all", "utterance", "time_stamp", "speaker", "sentence"]
        if args.output_format == 'one_tsv':
            args.select_data = ['file_id', 'spa_1', 'spa_2', 'spa_2a', 'speaker', 'sentence']
    spa_subtags = [x for x in args.select_data if re.search("spa_", x) is not None]
    if args.output_format == 'one_tsv':
        if not args.keep_empty_spa:
            # split for different files
            args.select_data = [[x for x in args.select_data if re.search("spa_", x) is None or x == tag] for tag in spa_subtags]
        else:
            args.select_data = [args.select_data]
            spa_subtags = ['spa_all']

    data = pd.read_csv(args.csv_loc)
    if not args.keep_empty_text:
        data.dropna(subset=["sentence"], inplace=True)
    if args.keep_empty_spa:
        args.ttv_split = [0., 1., 0.]
    ttv_rep = {f:np.random.choice([0,1,2], p=args.ttv_split) for f in data['file_id'].unique()}
    data["ttv"] = data["file_id"].apply(lambda x: ttv_rep[x])

    print(args)
    
    if args.replace_names:
        for col in ['sentence', 'lemmas']:
            # replace words with capital letter aka proper nouns
            data[col] = data[col].apply(lambda x: x if not isinstance(x, str) else ' '.join([w if (w.capitalize() != w) else replace_pnoun(w) for w in x.split()]))


    if args.output_format == 'sep_txt':
        ttv_writer = { 
                0: open(os.path.join('ttv', args.ttv_filepattern.format("train")), 'w'),
                1: open(os.path.join('ttv', args.ttv_filepattern.format("test")), 'w'),
                2: open(os.path.join('ttv', args.ttv_filepattern.format("valid")), 'w')
            }
        sep_loop(data, args.select_data, ttv_writer, remove_empty_tags=(not args.keep_empty_spa))
        for _,v in ttv_writer.items():
            v.close()
    
    elif args.output_format == 'one_tsv':
        ttv_writer = { 
            0: [os.path.join('ttv', args.ttv_filepattern.format("train_"+x)) for x in spa_subtags],
            1: [os.path.join('ttv', args.ttv_filepattern.format("test_"+x)) for x in spa_subtags],
            2: [os.path.join('ttv', args.ttv_filepattern.format("valid_"+x)) for x in spa_subtags]
        }
        # also duplicating line_function
        one_loop(data, args.select_data, ttv_writer, remove_empty_tags=(not args.keep_empty_spa))