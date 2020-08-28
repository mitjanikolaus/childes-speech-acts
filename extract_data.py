""" Formatting previously extracted data into datasets with correct format for training/testing

Examples:
    With training data (NewEngland corpus):
    >>> python extract_data.py --csv_loc formatted/NewEngland/data.csv --output_format one_tsv --ttv_filepattern childes_{}.tsv --select_data file_id spa_1 spa_2 spa_2a speaker sentence
    With test data (Bates corpus):
    >>> python extract_data.py --csv_loc formatted/Bates/data.csv --keep_empty_spa --output_format one_tsv --ttv_filepattern 'bates_{}.tsv'
"""
import json
import os, sys
from collections import Counter
import numpy as np
import pandas as pd
import time, datetime
import argparse
import re
from sklearn.utils import shuffle

from utils import dataset_labels, replace_pnoun

### Arguments
def argparser():
    """Create argparser, check arguments and return.
    """
    argparser = argparse.ArgumentParser(description='Extract and transform data.', formatter_class=argparse.RawTextHelpFormatter)
    # Data files
    argparser.add_argument('--csv_loc', type=str, default=None, help='location of the previously extracted data - csv file')
    argparser.add_argument('--json_loc', type=str, default=None, help='location of json files with extracted data - MACANNOT files')
    argparser.add_argument('--output_format', choices=['one_tsv', 'sep_txt'], help="""how to output the data based on the algorithm that will be applied next: 
        * 'one_tsv': HDF5 preparation for RNN
        * 'sep_txt': TXT preparation for CRF
    """)
    # Extraction parameters
    argparser.add_argument('--thres_replace', type=int, default=0, help="tags with a number of occurrences under this threshold will be replaced by the 'NEE' tag (Not Enough Examples)")
    argparser.add_argument('--labels_from', type=str, default=None, help="pattern of the other dataset to use as a reference for thres_replace")

    argparser.add_argument('--keep_untagged', action='store_true', help="whether to keep segments without spa tags (NOL) / with labels that don't meet the threshold (NEE) / with incorrect labels (NAT)")
    argparser.add_argument('--select_data', choices=["utterance", "spa_all", "spa_1", "spa_2", "spa_2a", "time_stamp", "speaker", "sentence", "lemmas", "pos", "action", "file_id", "age_months", "translation", "child"], nargs='+', help="ordered features for output - will be adapted to generate 3 files for ")
    # Actions on data
    argparser.add_argument('--replace_names', action='store_true', help="whether to replace locutors callings by tag; applicable on columns 'sentence' and 'lemmas'")
    argparser.add_argument('--match_age', type=int, nargs='+', default=None, help="ages to match data to - for split analysis")
    # Amount of data
    argparser.add_argument('--split_documents', choices=['no_split', 'on_blank', 'gaussian', 'exact'], default='no_split', help="whether to split the data into smaller documents; if any which splitting method to use")
    argparser.add_argument('--duplicate_documents', action="store_true", help="whether to split the data into smaller documents using a rolling window (increasing training data)")
    argparser.add_argument('--split_length', type=int, default=None, help="amount of lines (avg for gaussian) to put in split.")
    argparser.add_argument('--remove_seg_shorter_than', type=int, default=1, help="minimum amount of lines for a split to be considered.")
    # Train/test/validation
    argparser.add_argument('--ttv_split', nargs=3, type=float, default=[0.7, 0.25, 0.05], help="percentage of files going to train/test/validation sets; if --keep_empty_spa is False, test is all of the data")
    argparser.add_argument('--ttv_filepattern', type=str, default="list_{}_conv", help="file pattern (filled with train/test/valid + other patterns if need be) for ttv files - default 'list_{}_conv'.format(XXX)")
    argparser.add_argument('--unlabeled_data', action='store_true', help='for unlabeled data, no speech act is associated to sentences; cannot filter on empty tags (takes precedence over --ttv_split)')

    args = argparser.parse_args()
    if args.csv_loc is None and args.json_loc is None:
        raise ValueError("Data location must be given either through the --csv_loc or the --json_loc argument.")
    elif args.json_loc is not None and not os.path.isdir(args.json_loc):
        raise ValueError(f"{args.json_loc} is not a folder.")
    # not checking csv file, could have other extension but correct format

    if args.split_documents in ['gaussian', 'exact'] and args.split_length is None:
        raise ValueError("--split_length must be set")
    elif args.duplicate_documents and args.split_length is None:
        raise ValueError("--split_length must be set")

    if args.select_data is None:
        if args.output_format == 'sep_txt':
            args.select_data = ["spa_all", "utterance", "time_stamp", "speaker", "sentence"]
        if args.output_format == 'one_tsv':
            args.select_data = ['file_id', 'spa_1', 'spa_2', 'spa_2a', 'speaker', 'sentence']
    
    if args.unlabeled_data:
        args.ttv_split = [0., 1., 0.]

    if args.match_age is not None and "age_months" not in args.select_data:
        print("Adding 'age_months' column to output for consistency checks")
        args.select_data.append("age_months")

    return args


### WRITE DATA
def sep_loop(df:pd.DataFrame, line_function:list, ttv_writer:dict, remove_empty_tags:bool=True):
    # extract file <-> ttv columns, set as index, series to dict
    sets = df[['file_id', 'ttv']].drop_duplicates().set_index('file_id').to_dict()['ttv']
    # either remove empty tags or replace them with placeholder
    ss = [col for col in line_function if 'spa_' in col]
    if remove_empty_tags:
        for col in ss:
            # Note: hierarchical: all > 1 > 2 = 2a ; if one is empty the next are empty
            df.drop(df[df[col].isin(['NOL', 'NAT', 'NEE'])].index, inplace=True) 
    # No else, values are already replaced
    for k,v in sets.items():
        ttv_writer[v].write(k.replace('.json', '.txt')+'\n')
        df[df['file_id'] == k][line_function].to_csv(k.replace('.json', '.txt'), sep='\t', header=False, index=False)

def one_loop(df:pd.DataFrame, line_function:list, ttv_writer:dict, remove_empty_tags:bool=True):
    sub_data = {k: df[df['ttv'] == k] for k in range(0,3)}
    for k,v in ttv_writer.items():
        for columns, filepath in zip(line_function, v):
            tag = [col for col in columns if 'spa_' in col][0] # only supposed to be one tag by list of columns
            if remove_empty_tags:
                # Note: hierarchical: 1 > 2 = 2a ; if one is empty the next are empty
                sub_data[k].drop(sub_data[k][sub_data[k][tag].isin(['NOL', 'NAT', 'NEE'])].index, inplace=True) 
            sub_data[k][columns].rename(columns={col:col.upper() for col in columns}).to_csv(filepath, sep='\t', index=False)

# Create more data
def update_name(s:str, idx:int) -> str:
    """Add id to name, before extension (if exists).
    """
    s1 = s.split('/')
    s2 = s1[-1].split('.')
    if len(s2) > 1:
        s2 = '.'.join(s2[:-1]) + f'_{idx}' + '.' + s2[-1]
    else:
        s2 = s2[0] + f'_{idx}'
    s1[-1] = s2
    return '/'.join(s1)

def create_split_data(data:pd.DataFrame, conv_column:str, mode:str = 'exact', 
                    split_avg_lgth:int = 50, split_var_lgth:int=10, 
                    remove_seg_shorter_than:int=1, 
                    tag_column:str = None, empty_tags:list = ['NOL', 'NAT', 'NEE']) -> pd.DataFrame:
    """Split the data into segments of average length split_avg_lgth or on empty tags.
    column containing file references is updated to include splits.
    Splits too short are not included.
    """
    if mode not in ['gaussian', 'exact', 'on_blank']:
        raise ValueError(f"mode must be one of gaussian|exact|on_blank, currently {mode}")

    n = data.shape[0]
    if mode == "gaussian":
        split_index = np.cumsum(np.random.normal(split_avg_lgth, split_var_lgth, int(n/split_avg_lgth)).astype(int))
    elif mode == "exact":
        split_index = list(range(0,n,split_avg_lgth)) + [n] 
    elif mode == "on_blank":
        if tag_column is None:
            raise ValueError("name of column containing tags must be given.")
        split_index = list(data[data[tag_column].isin(empty_tags)].index)
    # update split index with data file changes index
    file_idx = data[data[conv_column] != data[conv_column].shift(1)].index.tolist()
    for f_idx in file_idx:
        split_index[min(range(len(split_index)), key = lambda i: abs(split_index[i]-f_idx))] = f_idx
    
    # split and shuffle
    tmp = []
    for i, idx in enumerate(split_index[:-1]):
        if (split_index[i+1]-idx) > remove_seg_shorter_than:
            subset = data.iloc[idx:split_index[i+1], :]
            subset[conv_column] = subset[conv_column].apply(lambda x: update_name(x,i))
            tmp.append(subset)
    tmp = shuffle(tmp)
    tmp = pd.concat(tmp, axis=0)

    # compute fraction and return
    return tmp

def create_rw_data(data:pd.DataFrame, conv_column:str, split_lgth:int = 50, remove_seg_shorter_than:int=10) -> pd.DataFrame:
    """Split the data into segments of length split_avg_lgth, and use a rolling window to create more training data. 
    column containing file references is updated to include splits, so that the algorithm doesn't later group data from one file all together despite splits.
    """
    # index of file change in data
    file_idx = data[data[conv_column] != data[conv_column].shift(1)].index.tolist()
    # create rolling windows
    rw_idx = [[(a, min(a+split_lgth, file_idx[i+1]-1)) for a in range(idx, file_idx[i+1])] for i, idx in enumerate(file_idx[:-1])]
    rw_idx = [y for x in rw_idx for y in x if not (y[1]-y[0] < remove_seg_shorter_than)] # flatten & removing sequences too short
    tmp = []
    for i, (idx_start, idx_end) in enumerate(rw_idx):
        subset = data.iloc[idx_start:idx_end, :].copy(deep=True) # otherwise conv_column is updated split_lgth times
        subset[conv_column] = subset[conv_column].apply(lambda x: update_name(x,i))
        tmp.append(subset)
    tmp = shuffle(tmp)
    tmp = pd.concat(tmp, axis=0)

    # return data
    return tmp

### READ DATA
def read_lines_from_folder(folder:str) -> pd.DataFrame:
    """Read all json files in the given folder, locate required fields and return data as pd.DataFrame
    """
    p = []
    for (root, dirs, files) in os.walk(folder):
        if dirs == []: # no more subfolders
            for file in sorted([x for x in files if x.split('.')[-1] == 'json']):
                json_fn = os.path.join(root, file)
                json_file = json.load(json_fn)
                for doc in json_file["documents"]:
                    # read all segments as sentence
                    sentence = " ".join([ x["word"] for x in doc["tokens"]])
                    d = {
                        'file_id': json_fn, 
                        "sentence": sentence, 
                        "time_stamp": "00:00:00" if "time" not in doc.keys() else doc["time"],
                        "spa_all": doc["segments"]["label"],
                        "speaker": doc["by"],
                        # add spa_1, spa_2, spa_2a
                        # pos = '' if "pos" not in doc["tokens"][0].keys() else " ".join([ x["pos"] for x in doc["tokens"]])
                        # lemmas = '' if "lemmas" not in doc["tokens"][0].keys() else " ".join([ x["lemmas"] for x in doc["tokens"]])
                        # action = ''
                    }
                p.append(d)
    return pd.DataFrame(p)

### UPDATE DATA
def replace_tag(x:str, tag_incorr:list, tag_toofew:list):
    if x == '' or x == 'NOL':
        return 'NOL'
    elif x in tag_incorr:
        return 'NAT'
    elif x in tag_toofew:
        return 'NEE'
    return x

# Update file containing metadata
def log_dataset(args):
    fname = 'ttv/db_metadata.json'
    entry = {}
    for arg in vars(args):
        entry[arg] = getattr(args, arg)
    # Dumping data
    a = []
    if not os.path.isfile(fname):
        a.append(entry)
        with open(fname, mode='w') as f:
            f.write(json.dumps(a, indent=2))
    else:
        with open(fname) as feedsjson:
            feeds = json.load(feedsjson)

        feeds.append(entry)
        with open(fname, mode='w') as f:
            f.write(json.dumps(feeds, indent=2))

def search_log(s:str) -> pd.Series:
    fname = 'ttv/db_metadata.json'
    with open(fname) as db:
        prev_dataset = json.load(db)
    p = pd.DataFrame(prev_dataset).set_index('ttv_filepattern')
    l = (p.index).tolist()
    if s not in l:
        raise ValueError(f"Cannot use pattern {s} as reference; existing patterns for data: {' '.join(l)}")
    return p.loc[s]

### MAIN
if __name__ == '__main__':
    args = argparser()
    print(args)

    spa_subtags = [x for x in args.select_data if re.search("spa_", x) is not None]
    translation_rename = ('translation' in args.select_data) and not ('sentence' in args.select_data)
    drop_subset = [col for col in ["sentence", "action"] if col in args.select_data]
    
    if args.output_format == 'one_tsv':
        # split for different files
        args.select_data = [[x for x in args.select_data if re.search("spa_", x) is None or x == tag] for tag in spa_subtags]
    
    if args.csv_loc is not None:
        data = pd.read_csv(args.csv_loc, keep_default_na=False) # can't have the 'NA' tag replaced by None! (can use "na_values = ''")
    else: # args.json_loc is not None
        data = read_lines_from_folder(args.json_loc)
    # Check translation
    if translation_rename:
        print("Renaming 'translation' column as 'sentence'")
        data["sentence"] = data["translation"]
    
    # Step 1: check labels
    if args.labels_from is not None:
        memory_args = search_log(args.labels_from)
    for spa_tag in spa_subtags:
        if spa_tag != ['spa_all']:
            labels = dataset_labels(spa_tag.upper())
            tag_counts = data[spa_tag].value_counts().to_dict()
            print(f'Tag {spa_tag} proportions:')
            print({k: np.round(v/data.shape[0],5) for k,v in tag_counts.items()})
            tag_incorr = [tag for tag in tag_counts.keys() if (tag not in labels.keys())]
            print('\tRemoving incorrect tags: ', ' '.join(tag_incorr))
            tag_toofew = [tag for tag, nbocc in tag_counts.items() if (nbocc < args.thres_replace)]
            if args.labels_from is not None:
                print(f"\tLoading tags to be removed (too few occurrences) from {args.labels_from}")
                tag_toofew = memory_args[spa_tag + '_toofew']
            print(f'\tRemoving tags with < {args.thres_replace} occurrences: ', ' '.join(tag_toofew))
            tag_kept = [tag for tag, nbocc in tag_counts.items() if (nbocc >= args.thres_replace) and (tag in labels.keys())]
            data[spa_tag] = data[spa_tag].fillna('NOL').apply(lambda x: replace_tag(x, tag_incorr, tag_toofew))
            # Saving statistics
            setattr(args, spa_tag + '_counts', dict(tag_counts))
            setattr(args, spa_tag + '_incorr', tag_incorr)
            setattr(args, spa_tag + '_toofew', tag_toofew)

    # Step 2: Update sentence/actions
    data[data["sentence"] =='.'] = ''
    # Step 3: If no sentence and no action and no translation: data is removed
    data = data[(pd.concat([data[col] != '' for col in drop_subset], axis=1)).any(axis=1)]

    # Step 4: Adapt data based on args (split_documents, duplicate_documents) - TODO
    if args.split_documents != "no_split":
        if len(spa_subtags) > 1 and args.split_documents == 'on_blank':
            raise ValueError("Cannot (yet) split data on blank when more than 1 tag is passed.")
        data = create_split_data(data, conv_column='file_id', mode=args.split_documents, 
                    split_avg_lgth=args.split_length, remove_seg_shorter_than=args.remove_seg_shorter_than)
    elif args.duplicate_documents:
        data = create_rw_data(data, conv_column='file_id', split_lgth=args.split_length, remove_seg_shorter_than=args.remove_seg_shorter_than)
    
    # Step 5: Group into train/test/validation (if need be)
    ttv_rep = {f:np.random.choice([0,1,2], p=args.ttv_split) for f in data['file_id'].unique()}
    data["ttv"] = data["file_id"].apply(lambda x: ttv_rep[x])
    
    if args.replace_names:
        for col in ['sentence', 'lemmas']:
            # replace words with capital letter aka proper nouns
            data[col] = data[col].apply(lambda x: x if not isinstance(x, str) else ' '.join([w if (w.capitalize() != w) else replace_pnoun(w) for w in x.split()]))
    if (args.match_age is not None) and ("age_months" in data.columns):
        match_age = args.match_age if isinstance(args.match_age, list) else [args.match_age]
        data['age_months'] = data.age_months.apply(lambda age: min(match_age, key=lambda x:abs(x-age)))

    if args.output_format == 'sep_txt':
        ttv_writer = { 
                0: open(os.path.join('ttv', args.ttv_filepattern.format("train")), 'w'),
                1: open(os.path.join('ttv', args.ttv_filepattern.format("test")), 'w'),
                2: open(os.path.join('ttv', args.ttv_filepattern.format("valid")), 'w')
            }
        sep_loop(data, args.select_data, ttv_writer, remove_empty_tags=(not args.keep_untagged))
        for _,v in ttv_writer.items():
            v.close()
    
    elif args.output_format == 'one_tsv':
        ttv_writer = { 
            i: [os.path.join('ttv', args.ttv_filepattern.format(fp+"_"+x)) for x in spa_subtags]
            for i, (fp, ratio) in enumerate(zip(['train', 'test', 'valid'], args.ttv_split)) if ratio > 0.
        }
        # also duplicating line_function
        one_loop(data, args.select_data, ttv_writer, remove_empty_tags=(not args.keep_untagged))
    
    # Logging execution:
    args.command = 'python ' + ' '.join(sys.argv)
    log_dataset(args)