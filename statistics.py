"""Training routine for LSTM and Transformer"""

import argparse
import os
import pickle
from collections import Counter

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np


def stats(args):
    # Load data
    vocab = pickle.load(open(args.data + "vocab.p", "rb"))
    label_vocab = pickle.load(open(args.data + "vocab_labels.p", "rb"))

    print("Loading data..")
    data = pd.read_hdf(args.data + os.sep + args.corpus, "test")

    labels = np.concatenate([np.array(l) for l in data.labels])
    labels = [label_vocab.inv[l] for l in labels]

    counter = Counter(labels).most_common()

    MIN_FREQUENCY = 100

    filtered = [(i, c) for i, c in counter if c >= MIN_FREQUENCY]

    plt.bar([l for l, c in filtered], [c for l, c in filtered])
    plt.show()

    print(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="./data/",
        help="location of the data corpus and vocabs",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="speech_acts_data_newengland.h5",
        help="name of the corpus file",
    )

    args = parser.parse_args()
    stats(args)
