import pickle
import argparse

from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score

from crf_annotate import calculate_frequencies
import matplotlib.pyplot as plt

import numpy as np

MIN_NUM_UTTERANCES = 200
THRESHOLD_ACQUIRED = 5


def parse_args():
    argparser = argparse.ArgumentParser()

    args = argparser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    ages_observed = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60] #, 66

    # Model prediction accuracies
    scores = pickle.load(open("data/classification_scores_crf.p", "rb"))
    scores_f1 = scores["f1-score"].to_dict()

    # Calculate overall adult speech act frequencies
    counts_aggregated = []
    for age in ages_observed:
        speech_acts_adult = pickle.load(open(f"data/speech_acts_adu_age_{age}.p", "rb"))
        counts_aggregated.extend(speech_acts_adult["y_pred"].tolist())

    frequencies_adults = calculate_frequencies(counts_aggregated)

    speech_acts_children = {}
    for age in ages_observed:
        speech_acts_children[age] = pickle.load(
            open(f"data/speech_acts_chi_age_{age}.p", "rb")
        )

    # Filter for speech acts that can be predicted with at least 60%
    observed_speech_acts = [k for k, v in scores_f1.items() if v > .6 and k in frequencies_adults.keys()]
    fraction_acquired_speech_act = {}
    age_of_acquisition = {}
    for speech_act in observed_speech_acts:
        fraction_acquired_speech_act[speech_act] = {}
        age_of_acquisition[speech_act] = max(ages_observed)
        for age in ages_observed:
            speech_acts_children_age = speech_acts_children[age]
            # TODO: fix: currently assuming a one-to-one mapping between file and child
            children_ids = speech_acts_children_age["file_id"].unique()
            n_children = 0
            n_acquired = 0
            for child_id in children_ids:
                speech_acts_child = speech_acts_children_age[speech_acts_children_age["file_id"] == child_id]
                if len(speech_acts_child) > MIN_NUM_UTTERANCES:
                    n_children += 1
                    target_speech_acts_child = speech_acts_child[
                        speech_acts_child["y_pred"] == speech_act
                    ]
                    if len(target_speech_acts_child) >= THRESHOLD_ACQUIRED:
                        n_acquired += 1

            fraction = n_acquired / n_children if n_children else 0

            fraction_acquired_speech_act[speech_act][age] = fraction

            if fraction > .5:
                age_of_acquisition[speech_act] = min(age, age_of_acquisition[speech_act])

        plt.plot(
            ages_observed,
            fraction_acquired_speech_act[speech_act].values(),
            label=speech_act,
        )

    plt.legend()


    frequencies_adults_observed = [frequencies_adults[s] for s in observed_speech_acts]

    # plt.scatter(age_of_acquisition.values(), frequencies_adults_observed)

    scores_observed = [scores_f1[s] for s in observed_speech_acts]
    # plt.scatter(age_of_acquisition.values(), scores_observed)

    features = np.array(frequencies_adults_observed).reshape(-1,1)
    targets = list(age_of_acquisition.values())
    reg = LinearRegression().fit(features, targets)
    y_pred = reg.predict(features)
    print("Explained variance (only freq):", explained_variance_score(targets, y_pred))

    features = np.array(list(zip(frequencies_adults_observed, scores_observed)))
    reg = LinearRegression().fit(features, targets)
    y_pred = reg.predict(features)
    print("Explained variance (freq + f1 scores):", explained_variance_score(targets, y_pred))





