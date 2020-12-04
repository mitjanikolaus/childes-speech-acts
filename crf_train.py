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
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
)
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite
from joblib import dump

### Tag functions
from preprocess import SPEECH_ACT, SPEAKER_ADULT
from utils import dataset_labels, select_tag


#### Read Data functions
def argparser():
    """Creating arparse.ArgumentParser and returning arguments"""
    argparser = argparse.ArgumentParser(
        description="Train a CRF and test it.",
    )
    # Data files
    argparser.add_argument("train", type=str, help="file listing train dialogs")
    # Operations on data
    argparser.add_argument(
        "--match_age",
        type=int,
        nargs="+",
        default=None,
        help="ages to match data to - for split analysis",
    )
    # parameters for training:
    argparser.add_argument(
        "--nb_occurrences",
        "-noc",
        type=int,
        default=5,
        help="number of minimum occurrences for word to appear in features",
    )
    argparser.add_argument(
        "--use_action",
        "-act",
        action="store_true",
        help="whether to use action features to train the algorithm, if they are in the data",
    )
    argparser.add_argument(
        "--use_past",
        "-past",
        action="store_true",
        help="whether to add previous sentence as features",
    )
    argparser.add_argument(
        "--use_repetitions",
        "-rep",
        action="store_true",
        help="whether to check in data if words were repeated from previous sentence, to train the algorithm",
    )
    argparser.add_argument(
        "--use_past_actions",
        "-pa",
        action="store_true",
        help="whether to add actions from the previous sentence to features",
    )
    argparser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to display training iterations output.",
    )

    args = argparser.parse_args()

    return args


#### Features functions
def add_feature_columns(
    p: pd.DataFrame,
    match_age: Union[str, list] = None,
    use_action: bool = False,
    use_past: bool = False,
    use_pastact: bool = False,
    check_repetition: bool = False,
):
    """Function adding features to the data:
    * turn_length
    * tags (if necessary): extract interchange/illocutionary from general tag
    * action_tokens (if necessary): splitting action sentence into individual words
    * age_months: matching age to experimental labels
    * repeted_words:
    * number of repeated words
    * ratio of words that were repeated from previous sentence over sentence length
    """
    # sentence: using tokens to count & all
    p["tokens"] = p.tokens
    p["turn_length"] = p.tokens.apply(len)

    # action: creating action tokens
    if use_action:
        p["action"].fillna("", inplace=True)
        p["action_tokens"] = p.action.apply(lambda x: x.lower().split())

    # matching age with theoretical age from the study
    if "age_months" in p.columns and match_age is not None:
        match_age = match_age if isinstance(match_age, list) else [match_age]
        p["age_months"] = p.age_months.apply(
            lambda age: min(match_age, key=lambda x: abs(x - age))
        )

    # repetition features
    if check_repetition or use_past or use_pastact:
        p["prev_file"] = p.file_id.shift(1).fillna(p.file_id.iloc[0])
        p["prev_spk"] = p.speaker.shift(1).fillna(p.speaker.iloc[0])
        p["prev_st"] = p.tokens.shift(
            1
        )
        p["prev_st"].iloc[0] = p.tokens.iloc[0]

    if check_repetition:
        p["repeated_words"] = p.apply(
            lambda x: [w for w in x.tokens if w in x.prev_st]
            if (x.prev_spk != x.speaker) and (x.file_id == x.prev_file)
            else [],
            axis=1,
        )
        p["nb_repwords"] = p.repeated_words.apply(len)
        p["ratio_repwords"] = p.nb_repwords / p.turn_length

    if use_past:
        p["past"] = p.apply(
            lambda x: x.prev_st if (x.file_id == x.prev_file) else [], axis=1
        )

    if use_action and use_pastact:
        p["prev_act"] = p["action_tokens"].shift(1)
        p["prev_act"].iloc[0] = p["action_tokens"].iloc[0]
        p["past_act"] = p.apply(
            lambda x: x.prev_act if (x.file_id == x.prev_file) else [], axis=1
        )
    # remove helper columns
    p = p.drop(columns=["prev_spk", "prev_st", "prev_file", "prev_act"], errors="ignore")

    # return Dataframe
    return p


def get_features_from_row(
    features: dict, spoken_tokens: list, speaker: str, ln: int, **kwargs
):
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
    feat_glob = {
        "words": Counter([w for w in spoken_tokens if (w in features["words"].keys())])
    }  # TODO: add 'UNK' token
    feat_glob["speaker"] = {speaker: 1.0}
    feat_glob["speaker"] = 1.0 if speaker == SPEAKER_ADULT else 0.0
    feat_glob["length"] = {
        k: (1 if ln <= float(k.split("-")[1]) and ln >= float(k.split("-")[0]) else 0)
        for k in features["length_bins"].keys()
    }

    if ("action_tokens" in kwargs) and (kwargs["action_tokens"] is not None):
        # actions are descriptions just like 'words'
        feat_glob["actions"] = Counter(
            [w for w in kwargs["action_tokens"] if (w in features["action"].keys())]
        )  # if (features['action'] is not None) else Counter(action_tokens)
    if ("repetitions" in kwargs) and (kwargs["repetitions"] is not None):
        (rep_words, len_rep, ratio_rep) = kwargs["repetitions"]
        feat_glob["repeated_words"] = Counter(
            [w for w in rep_words if (w in features["words"].keys())]
        )
        feat_glob["rep_length"] = {
            k: (
                1
                if len_rep <= float(k.split("-")[1])
                and len_rep >= float(k.split("-")[0])
                else 0
            )
            for k in features["rep_length_bins"].keys()
        }
        feat_glob["rep_ratio"] = {
            k: (
                1
                if ratio_rep <= float(k.split("-")[1])
                and ratio_rep >= float(k.split("-")[0])
                else 0
            )
            for k in features["rep_ratio_bins"].keys()
        }
    if ("past_tokens" in kwargs) and (kwargs["past_tokens"] is not None):
        feat_glob["past"] = Counter(
            [w for w in kwargs["past_tokens"] if (w in features["words"].keys())]
        )
    if ("pastact_tokens" in kwargs) and (kwargs["pastact_tokens"] is not None):
        feat_glob["past_actions"] = Counter(
            [w for w in kwargs["pastact_tokens"] if (w in features["action"].keys())]
        )

    return feat_glob


def word_bs_feature(
    features: dict, spoken_tokens: list, speaker: str, ln: int, **kwargs
):
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
    nb_features = (
        max([max([int(x) for x in v.values()]) for v in features.values()]) + 1
    )
    # list features
    features_sparse = [
        features["words"][w] for w in spoken_tokens if w in features["words"].keys()
    ]  # words
    features_sparse.append(features["speaker"][speaker])  # locutor
    for k in features["length_bins"].keys():  # length
        if ln <= float(k.split("-")[1]) and ln >= float(k.split("-")[0]):
            features_sparse.append(features["length_bins"][k])

    if ("action_tokens" in kwargs) and (
        kwargs["action_tokens"] is not None
    ):  # actions are descriptions just like 'words'
        features_sparse += [
            features["action"][w]
            for w in kwargs["action_tokens"]
            if w in features["action"].keys()
        ]
    if ("repetitions" in kwargs) and (
        kwargs["repetitions"] is not None
    ):  # not using words, only ratio+len
        (_, len_rep, ratio_rep) = kwargs["repetitions"]
        for k in features["rep_length_bins"].keys():
            if len_rep <= float(k.split("-")[1]) and len_rep >= float(k.split("-")[0]):
                features_sparse.append(features["rep_length_bins"][k])
        for k in features["rep_ratio_bins"].keys():
            if len_rep <= float(k.split("-")[1]) and len_rep >= float(k.split("-")[0]):
                features_sparse.append(features["rep_ratio_bins"][k])

    # transforming features
    features_full = [1 if i in features_sparse else 0 for i in range(nb_features)]

    return features_full


def generate_features(
    data: pd.DataFrame,
    nb_occ: int,
    use_action: bool,
    use_repetitions: bool,
    bin_cut: int = 10,
) -> dict:
    """Analyse data according to arguments passed and generate features_idx dictionary. Printing log data to console."""
    print("\nTag counts: ")
    count_tags = data[SPEECH_ACT].value_counts().to_dict()
    for k in sorted(count_tags.keys()):
        print("{}: {}".format(k, count_tags[k]), end=" ; ")

    # Features: vocabulary (spoken)
    count_vocabulary = [y for x in data.tokens.tolist() for y in x]
    count_vocabulary = dict(Counter(count_vocabulary))
    count_vocabulary = {k: v for k, v in count_vocabulary.items() if v > nb_occ}

    # turning vocabulary into numbered features - ordered vocabulary
    feature_vocabs = {
        "words": {k: i for i, k in enumerate(sorted(count_vocabulary.keys()))}
    }
    print("\nThere are {} words in the vocab".format(len(feature_vocabs["words"])))

    # Features: Speakers (+ logging counts)
    # count_spk = dict(Counter(data["speaker"].tolist()))
    # print("\nSpeaker counts: ")
    # for k in sorted(count_spk.keys()):
    #     print("{}: {}".format(k, count_spk[k]), end=" ; ")
    # feature_vocabs["speaker"] = {
    #     k: (len(feature_vocabs["words"]) + i)
    #     for i, k in enumerate(sorted(count_spk.keys()))
    # }

    # Features: sentence length (+ logging counts)
    data["len_bin"], bins = pd.qcut(
        data.turn_length, q=bin_cut, duplicates="drop", labels=False, retbins=True
    )
    print("\nTurn length splits: ")
    for i, k in enumerate(bins[:-1]):
        print("\tlabel {}: turns of length {}-{}".format(i, k, bins[i + 1]))

    nb_feat = max([max(v.values()) for v in feature_vocabs.values()])
    feature_vocabs["length_bins"] = {
        "{}-{}".format(k, bins[i + 1]): (nb_feat + i) for i, k in enumerate(bins[:-1])
    }
    feature_vocabs["length"] = {i: (nb_feat + i) for i, _ in enumerate(bins[:-1])}
    # parameters: duplicates: 'raise' raises error if bins are identical, 'drop' just ignores them (leading to the creation of larger bins by fusing those with identical cuts)
    # retbins = return bins (for debug) ; labels=False: only yield the position in the binning, not the name (simpler to create features)

    # Features: actions
    if use_action:
        count_actions = [y for x in data.action_tokens.tolist() for y in x]  # flatten
        count_actions = dict(Counter(count_actions))
        # filtering features
        count_actions = {k: v for k, v in count_actions.items() if v > nb_occ}
        # turning vocabulary into numbered features - ordered vocabulary
        nb_feat = max([max(v.values()) for v in feature_vocabs.values()])
        feature_vocabs["action"] = {
            k: i + nb_feat for i, k in enumerate(sorted(count_actions.keys()))
        }
        print("\nThere are {} words in the actions".format(len(feature_vocabs["action"])))

    # Features: repetitions (reusing word from speech)
    if use_repetitions:
        nb_feat = max([max(v.values()) for v in feature_vocabs.values()])
        # features esp for length & ratio - repeated words can use previously defined features
        # lengths
        _, bins = pd.qcut(
            data.nb_repwords, q=bin_cut, duplicates="drop", labels=False, retbins=True
        )
        feature_vocabs["rep_length_bins"] = {
            "{}-{}".format(k, bins[i + 1]): (nb_feat + i)
            for i, k in enumerate(bins[:-1])
        }
        # ratios
        _, bins = pd.qcut(
            data.ratio_repwords,
            q=bin_cut,
            duplicates="drop",
            labels=False,
            retbins=True,
        )
        feature_vocabs["rep_ratio_bins"] = {
            "{}-{}".format(k, bins[i + 1]): (nb_feat + i)
            for i, k in enumerate(bins[:-1])
        }
        print("\nRepetition ratio splits: ")
        for i, k in enumerate(bins[:-1]):
            print("\tlabel {}: turns of length {}-{}".format(i, k, bins[i + 1]))

    return feature_vocabs


### REPORT
def plot_training(trainer, file_name):
    logs = pd.DataFrame(trainer.logparser.iterations)  # initially list of dicts
    # columns: {'loss', 'error_norm', 'linesearch_trials', 'active_features', 'num', 'time', 'scores', 'linesearch_step', 'feature_norm'}
    # FYI scores is empty

    logs.set_index("num", inplace=True)
    for col in ["loss", "active_features"]:
        plt.figure()
        plt.plot(logs[col])
        plt.savefig(file_name + "/" + col + ".png")


#### Check predictions
def crf_predict(
    tagger: pycrfsuite.Tagger,
    gp_data: list,
    mode: str = "raw",
    exclude_labels: list = ["NOL", "NAT", "NEE"],
) -> Union[list, Tuple[list, pd.DataFrame]]:
    """Return predictions for the test data, grouped by file. 3 modes for return:
            * Return raw predictions (raw)
            * Return predictions with only valid tags (exclude_ool)
            * Return predictions (valid tags) and probabilities for each class (rt_proba)

    Predictions are returned unflattened

    https://python-crfsuite.readthedocs.io/en/latest/pycrfsuite.html
    """
    if mode not in ["raw", "exclude_ool", "rt_proba"]:
        raise ValueError(
            f"mode must be one of raw|exclude_ool|rt_proba; currently {mode}"
        )
    if mode == "raw":
        return [tagger.tag(xseq) for xseq in gp_data]
    labels = tagger.labels()

    res = []
    y_pred = []
    for fi, xseq in enumerate(gp_data):
        tagger.set(xseq)
        file_proba = pd.DataFrame(
            {
                label: [tagger.marginal(label, i) for i in range(len(xseq))]
                for label in labels
            }
        )
        y_pred.append(
            file_proba[[col for col in file_proba.columns if col not in exclude_labels]]
            .idxmax(axis=1)
            .tolist()
        )
        file_proba["file_id"] = fi
        res.append(file_proba)

    if mode == "rt_proba":
        return y_pred, pd.concat(res, axis=0)
    return y_pred  # else


#### MAIN
if __name__ == "__main__":
    args = argparser()
    print(args)

    # Definitions
    number_words_for_feature = args.nb_occurrences  # default 5
    number_segments_length_feature = 10
    # number_segments_turn_position = 10 # not used for now

    print("### Loading data:".upper())

    data_train = pd.read_pickle(args.train)

    args.use_action = args.use_action & ("action" in data_train.columns.str.lower())
    args.use_past_actions = args.use_past_actions & args.use_action
    data_train = add_feature_columns(
        data_train,
        use_action=args.use_action,
        match_age=args.match_age,
        check_repetition=args.use_repetitions,
        use_past=args.use_past,
        use_pastact=args.use_past_actions,
    )

    print("### Creating features:")
    features_idx = generate_features(
        data_train,
        args.nb_occurrences,
        args.use_action,
        args.use_repetitions,
        bin_cut=number_segments_length_feature,
    )

    # creating crf features set for train
    data_train["features"] = data_train.apply(
        lambda x: get_features_from_row(
            features_idx,
            x.tokens,
            x["speaker"],
            x.turn_length,
            action_tokens=None if not args.use_action else x.action_tokens,
            repetitions=None
            if not args.use_repetitions
            else (x.repeated_words, x.nb_repwords, x.ratio_repwords),
            past_tokens=None if not args.use_past else x.past,
            pastact_tokens=None if not args.use_past_actions else x.past_act,
        ),
        axis=1,
    )

    # Once the features are done, groupby name and extract a list of lists
    # some "None" appear bc some illocutionary codes missing - however creating separations between data...
    grouped_train = (
        data_train.dropna(subset=[SPEECH_ACT])
        .groupby(by=["file_id"])
        .agg(
            {
                "features": lambda x: [y for y in x],
                SPEECH_ACT: lambda x: [y for y in x],
                "index": min,
            }
        )
    )  # listed by apparition order
    grouped_train = sklearn.utils.shuffle(grouped_train)

    # After that, train ---
    print("\n### Training starts.".upper())
    trainer = pycrfsuite.Trainer(verbose=args.verbose)
    # Adding data
    for idx, file_data in grouped_train.iterrows():
        trainer.append(file_data["features"], file_data[SPEECH_ACT])  # X_train, y_train
    # Parameters
    trainer.set_params(
        {
            "c1": 1,  # coefficient for L1 penalty
            "c2": 1e-3,  # coefficient for L2 penalty
            "max_iterations": 50,  # stop earlier
            "feature.possible_transitions": True,  # include transitions that are possible, but not observed
        }
    )

    # Location for weight save
    name = os.path.join(
        os.getcwd(),
        "_".join(
            [
                x
                for x in [
                    SPEECH_ACT,
                    datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"),
                ]
                if x
            ]
        ),
    )  # creating name with arguments, removing Nones in list
    print("Saving model at: {}".format(name))
    os.mkdir(name)
    trainer.train(os.path.join(name, "model.pycrfsuite"))
    # plotting training curves
    # plot_training(trainer, name)
    # dumping features
    with open(os.path.join(name, "features.json"), "w") as json_file:
        json.dump(features_idx, json_file)
    # dumping metadata
    with open(os.path.join(name, "metadata.txt"), "w") as meta_file:
        for arg in vars(args):
            meta_file.write("{0}:\t{1}\n".format(arg, getattr(args, arg)))

    tagger = pycrfsuite.Tagger()
    tagger.open(os.path.join(name, "model.pycrfsuite"))
    y_pred = crf_predict(
        tagger,
        grouped_train.sort_values("index", ascending=True)["features"],
        mode="exclude_ool",
    )
    data_train["y_pred"] = [y for x in y_pred for y in x]  # flatten

    # TODO check whether we do similar filter for LSTM eval!
    data_crf = data_train[~data_train[SPEECH_ACT].isin(["NOL", "NAT", "NEE"])]
    acc = accuracy_score(
        data_crf[SPEECH_ACT].tolist(), data_crf["y_pred"].tolist(), normalize=True
    )
    print(acc)
