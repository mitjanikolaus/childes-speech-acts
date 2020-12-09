import pickle
import argparse
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score

from crf_annotate import calculate_frequencies
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from utils import COLORS_PLOT_CATEGORICAL

MIN_NUM_UTTERANCES = 500
MIN_CHILDREN_REQUIRED = 3
THRESHOLD_ACQUIRED = 1
AGE_MONTHS_BIN_SIZE = 1
MIN_AGE = 12
MAX_AGE = 60


if __name__ == "__main__":
    # Model prediction accuracies
    print("Loading data...")
    scores = pickle.load(open("data/classification_scores_crf.p", "rb"))
    scores_f1 = scores["f1-score"].to_dict()

    # Calculate overall adult speech act frequencies
    speech_acts_adult = pd.read_hdf("data/speech_acts_adu.h5")
    counts_adults = speech_acts_adult["y_pred"].to_list()
    frequencies_adults = calculate_frequencies(counts_adults)

    # Filter for speech acts that can be predicted with at least 60%
    # TODO justify
    observed_speech_acts = [
        k for k, v in scores_f1.items() if v > 0.6 and k in frequencies_adults.keys()
    ]

    # dev: use only subset
    # observed_speech_acts = observed_speech_acts[:20]

    speech_acts_children = pd.read_hdf("data/speech_acts_chi.h5")

    fraction_acquired_speech_act = []

    print("Processing speech acts...")
    for speech_act in observed_speech_acts:
        prev_fraction = 0
        for month in range(MIN_AGE, MAX_AGE, AGE_MONTHS_BIN_SIZE):
            speech_acts_children_month = speech_acts_children[
                (speech_acts_children["child_age"] >= month)
                & (speech_acts_children["child_age"] < month + AGE_MONTHS_BIN_SIZE)
            ]
            children_ids = speech_acts_children_month["child_id"].unique()
            n_children = 0
            n_acquired = 0
            for child_id in children_ids:
                speech_acts_child = speech_acts_children_month[
                    speech_acts_children_month["child_id"] == child_id
                ]
                if len(speech_acts_child) > MIN_NUM_UTTERANCES:
                    n_children += 1
                    target_speech_acts_child = speech_acts_child[
                        speech_acts_child["y_pred"] == speech_act
                    ]
                    if len(target_speech_acts_child) >= THRESHOLD_ACQUIRED:
                        n_acquired += 1

            if n_children >= MIN_CHILDREN_REQUIRED:
                fraction = n_acquired / n_children
            else:
                # not enough data, use data of previous month
                warnings.warn(
                    f"speech act {speech_act}: month {month}: Not enough data (only {n_children} children). Using value of previous month. Increase age bin size?"
                )
                fraction = prev_fraction
            # print(f"{speech_act}: {month}: {n_children}")

            fraction_acquired_speech_act.append(
                {
                    "speech_act": speech_act,
                    "month": month,
                    "fraction": fraction,
                }
            )
            prev_fraction = fraction

    fraction_acquired_speech_act = pd.DataFrame(fraction_acquired_speech_act)

    frequencies_adults_observed = [frequencies_adults[s] for s in observed_speech_acts]

    scores_observed = [scores_f1[s] for s in observed_speech_acts]

    sns.set_palette(COLORS_PLOT_CATEGORICAL)
    g = sns.lmplot(
        data=fraction_acquired_speech_act,
        x="month",
        y="fraction",
        hue="speech_act",
        logistic=True,
        ci=None,
        legend=True,
    )
    g.set(ylim=(0, 1))
    plt.setp(g.legend.get_texts(), fontsize="10")
    # plt.legend(fontsize='20', title_fontsize='40', loc)

    # Read estimated ages of acquisition from the logistic regression plot data
    age_of_acquisition = {}
    for i, speech_act in enumerate(observed_speech_acts):
        fractions = g.ax.get_lines()[i].get_ydata()
        ages = g.ax.get_lines()[i].get_xdata()
        if np.where(fractions > 0.5)[0].size > 0:
            age_of_acquisition[speech_act] = ages[np.min(np.where(fractions > 0.5))]
        else:
            age_of_acquisition[speech_act] = MAX_AGE
        print(
            f"Age of acquisition of {speech_act}: {age_of_acquisition[speech_act]:.1f}"
        )

    features = np.array(frequencies_adults_observed).reshape(-1, 1)
    targets = list(age_of_acquisition.values())
    reg = LinearRegression().fit(features, targets)
    y_pred = reg.predict(features)
    print("Explained variance (only freq):", explained_variance_score(targets, y_pred))

    features = np.array(list(zip(frequencies_adults_observed, scores_observed)))
    reg = LinearRegression().fit(features, targets)
    y_pred = reg.predict(features)
    print(
        "Explained variance (freq + f1 scores):",
        explained_variance_score(targets, y_pred),
    )

    plt.show()
