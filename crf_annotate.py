import os
import pickle
import sys
import random
import codecs
import argparse
import time, datetime
from collections import Counter
import json
import ast
from typing import Union, Tuple
from bidict import bidict

import seaborn as sns

from childespy.childespy import get_transcripts, get_corpora, get_utterances

import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from scipy.stats import entropy
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite
from joblib import load

### Tag functions
from tqdm import tqdm

from crf_test import crf_predict
from utils import dataset_labels, check_tag_pattern
from crf_train import openData, data_add_features, word_to_feature, word_bs_feature

#### Read Data functions
def parse_args():
    argparser = argparse.ArgumentParser(description="Annotate.")
    # Data files
    argparser.add_argument(
        "--model",
        "-m",
        required=True,
        type=str,
        default=None,
        help="folder containing model, features and metadata",
    )
    argparser.add_argument(
        "--data", type=str, required=True, help="Path to preprocessed data to annotate"
    )
    argparser.add_argument(
        "--compare", type=str, required=True, help="Path to frequencies to compare to"
    )

    args = argparser.parse_args()

    return args



if __name__ == "__main__":
    args = parse_args()

    # Loading data
    data = pickle.load(open(args.data, "rb"))

    # Loading model
    model_path = args.model + os.sep + "model.pycrfsuite"
    features_path = args.model + os.sep + "features.json"

    data = pd.DataFrame(data).reset_index(drop=False)

    # Add turn length column
    data["turn_length"] = data.tokens.apply(len)

    # Replace speaker column values
    data["speaker"] = data["speaker"].apply(
        lambda x: "CHI" if x == "Target_Child" else "MOT"
    )

    # Loading features
    with open(features_path, "r") as json_file:
        features_idx = json.load(json_file)

    # TODO use more features?
    data["features"] = data.apply(
        lambda x: word_to_feature(
            features_idx,
            x.tokens,
            x["speaker"],
            x.turn_length,
        ),
        axis=1,
    )

    # Predictions
    tagger = pycrfsuite.Tagger()
    tagger.open(model_path)

    # creating data
    X_dev = data.groupby(by=["file_id"]).agg(
        {"features": lambda x: [y for y in x], "index": min}
    )
    y_pred = crf_predict(
        tagger,
        X_dev.sort_values("index", ascending=True)["features"],
    )
    data["y_pred"] = [y for x in y_pred for y in x]

    # Filter for children's utterances
    data_children = data[data.speaker == "CHI"]

    counts = Counter(data_children["y_pred"].tolist())
    for k in counts.keys():
        if counts[k]:
            counts[k] /= len(data_children)
        else:
            counts[k] = 0

    gold_frequencies = pickle.load(open(args.compare, "rb"))
    counts = {k: counts[k] for k in gold_frequencies.keys()}

    kl_divergence = entropy(
        list(counts.values()), qk=list(gold_frequencies.values())
    )
    print(f"KL Divergence: {kl_divergence:.3f}")

    labels = list(gold_frequencies.keys()) * 2
    source = ["Gold"] * len(gold_frequencies) + ["Predicted"] * len(gold_frequencies)
    counts = list(gold_frequencies.values()) + list(counts.values())
    df = pd.DataFrame(zip(labels, source, counts), columns=["speech_act", "source", "frequency"])
    plt.figure(figsize=(10, 6))
    sns.barplot(x="speech_act", hue="source", y="frequency", data=df)
    plt.title(f"{args.data} compared to {args.compare} | KL Divergence: {kl_divergence:.3f}")
    plt.show()

    #
    # for _, row in data.iterrows():
    #     print(f"({row.y_pred}) {row.speaker}: {' '.join(row.tokens)}")

