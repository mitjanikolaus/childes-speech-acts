import argparse
import pickle

from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score

from age_of_acquisition import TARGET_PRODUCTION, TARGET_COMPREHENSION, MAX_AGE
from crf_annotate import calculate_frequencies
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from preprocess import SPEECH_ACT, CHILD

if __name__ == "__main__":
    print("Loading data...")
    # Calculate overall adult speech act frequencies
    data = pd.read_pickle("data/new_england_preprocessed.p")

    frequencies = calculate_frequencies(data[SPEECH_ACT])
    frequencies = dict(frequencies.most_common())

    plt.bar(list(frequencies.keys()), list(frequencies.values()))
    plt.xlabel("speech act")
    plt.ylabel("frequency")
    plt.xticks([])

    plt.show()
