import argparse
import pickle

from scipy.stats import pearsonr
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score

from age_of_acquisition import MAX_AGE, TARGET_PRODUCTION, TARGET_COMPREHENSION
from crf_annotate import calculate_frequencies
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from preprocess import SPEECH_ACT, CHILD
from utils import COLLAPSED_FORCE_CODES_TRANSLATIONS

if __name__ == "__main__":
    print("Loading data...")
    ages_of_acquisition_production = pickle.load(
        open(f"results/age_of_acquisition_production_collapsed.p", "rb")
    )
    ages_of_acquisition_comprehension = pickle.load(
        open(f"results/age_of_acquisition_comprehension_collapsed.p", "rb")
    )

    observed_speech_acts = list(ages_of_acquisition_production.keys())

    # Filter out speech acts that do occur in comprehension data
    observed_speech_acts = [s for s in observed_speech_acts if s in ages_of_acquisition_comprehension]

    print("Number of speech acts analyzed: ", len(observed_speech_acts))

    ages_of_acquisition_production = [ages_of_acquisition_production[s] for s in observed_speech_acts]
    ages_of_acquisition_comprehension = [ages_of_acquisition_comprehension[s] for s in observed_speech_acts]

    fig, ax = plt.subplots()
    x = ages_of_acquisition_production
    y = ages_of_acquisition_comprehension
    plt.scatter(x, y)
    plt.xlabel(f"Production: age of acquisition (months)")
    plt.ylabel(f"Comprehension: age of acquisition (months)")
    plt.xlim(14, 60)
    plt.ylim(14, 60)

    for i, speech_act in enumerate(observed_speech_acts):
        ax.annotate(speech_act, (x[i], y[i]))

    pearson_r = pearsonr(x, y)
    print("Pearson r for production vs. comprehension: ", pearson_r)

    plt.show()
