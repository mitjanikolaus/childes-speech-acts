import pickle
import argparse

from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score

from crf_annotate import calculate_frequencies
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

MIN_NUM_UTTERANCES = 500
THRESHOLD_ACQUIRED = 1


def parse_args():
    argparser = argparse.ArgumentParser()

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    min_age = 12
    max_age = 60

    # Model prediction accuracies
    scores = pickle.load(open("data/classification_scores_crf.p", "rb"))
    scores_f1 = scores["f1-score"].to_dict()

    # Calculate overall adult speech act frequencies
    speech_acts_adult = pd.read_hdf("data/speech_acts_adu.h5")
    counts_adults = speech_acts_adult["y_pred"].to_list()
    frequencies_adults = calculate_frequencies(counts_adults)

    # Filter for speech acts that can be predicted with at least 60%
    observed_speech_acts = [k for k, v in scores_f1.items() if v > .6 and k in frequencies_adults.keys()]

    # dev: use only subset
    observed_speech_acts = observed_speech_acts[:10]

    speech_acts_children = pd.read_hdf("data/speech_acts_chi.h5")

    fraction_acquired_speech_act = []

    age_of_acquisition = {}

    for speech_act in observed_speech_acts:
        age_of_acquisition[speech_act] = max_age
        for month in range(min_age, max_age):
            prev_fraction = 0
            speech_acts_children_month = speech_acts_children[speech_acts_children["child_age"] == month]
            children_ids = speech_acts_children_month["child_id"].unique()
            n_children = 0
            n_acquired = 0
            for child_id in children_ids:
                speech_acts_child = speech_acts_children_month[speech_acts_children_month["child_id"] == child_id]
                if len(speech_acts_child) > MIN_NUM_UTTERANCES:
                    n_children += 1
                    target_speech_acts_child = speech_acts_child[
                        speech_acts_child["y_pred"] == speech_act
                    ]
                    if len(target_speech_acts_child) >= THRESHOLD_ACQUIRED:
                        n_acquired += 1

            if n_children > 5:
                fraction = n_acquired / n_children
            else:
                # not enough data, use data of previous month
                # TODO probably better increase bin size..
                fraction = prev_fraction
            print(f"{speech_act}: {month}: {n_children}")

            fraction_acquired_speech_act.append({
                "speech_act": speech_act,
                "month": month,
                "fraction": fraction,
            })
            prev_fraction = fraction

            if fraction > .5:
                age_of_acquisition[speech_act] = min(month, age_of_acquisition[speech_act])

    fraction_acquired_speech_act = pd.DataFrame(fraction_acquired_speech_act)

    sns.set_palette("tab20")
    sns.lineplot(data = fraction_acquired_speech_act, x = "month", y="fraction", hue="speech_act")
    # sns.lmplot(data = fraction_acquired_speech_act, x = "month", y="fraction", hue="speech_act")



    frequencies_adults_observed = [frequencies_adults[s] for s in observed_speech_acts]

    scores_observed = [scores_f1[s] for s in observed_speech_acts]

    features = np.array(frequencies_adults_observed).reshape(-1,1)
    targets = list(age_of_acquisition.values())
    reg = LinearRegression().fit(features, targets)
    y_pred = reg.predict(features)
    print("Explained variance (only freq):", explained_variance_score(targets, y_pred))

    features = np.array(list(zip(frequencies_adults_observed, scores_observed)))
    reg = LinearRegression().fit(features, targets)
    y_pred = reg.predict(features)
    print("Explained variance (freq + f1 scores):", explained_variance_score(targets, y_pred))





