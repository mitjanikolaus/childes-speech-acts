import pickle
import warnings

from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score

from crf_annotate import calculate_frequencies
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from preprocess import SPEECH_ACT
from utils import COLORS_PLOT_CATEGORICAL, age_bin

MIN_NUM_UTTERANCES = 0
MIN_CHILDREN_REQUIRED = 0
THRESHOLD_ACQUIRED = 1
THRESHOLD_FRACTION_ACQUIRED = 0.5

def get_fraction_contingent_responses(ages, observed_speech_acts):
    """Calculate "understanding" of speech acts by measuring the amount of contingent responses"""
    fraction_contingent_responses = []

    for month in ages:
        contingency_data = pd.read_csv(f"adjacency_pairs/ADU-CHI_age_{month}_contingency.csv")

        for speech_act in observed_speech_acts:
            # Add start: at 10 months children don't produce any speech act
            fraction_contingent_responses.append(
                {
                    "speech_act": speech_act,
                    "month": 10,
                    "fraction": 0.0,
                }
            )
            # Add end: at 18 years children know all speech acts
            fraction_contingent_responses.append(
                {
                    "speech_act": speech_act,
                    "month": 12 * 18,
                    "fraction": 1.0,
                }
            )

            fraction = contingency_data[(contingency_data["source"] == speech_act) & (contingency_data["contingency"] == 1)]["fraction"].sum()

            fraction_contingent_responses.append(
                {
                    "speech_act": speech_act,
                    "month": month,
                    "fraction": fraction,
                }
            )

    return pd.DataFrame(fraction_contingent_responses)


def get_fraction_producing_speech_acts(data_children, ages, observed_speech_acts):
    fraction_acquired_speech_act = []

    print("Processing speech acts...")
    for speech_act in observed_speech_acts:

        # Add start: at 10 months children don't produce any speech act
        fraction_acquired_speech_act.append(
            {
                "speech_act": speech_act,
                "month": 10,
                "fraction": 0.0,
            }
        )
        # Add end: at 18 years children know all speech acts
        fraction_acquired_speech_act.append(
            {
                "speech_act": speech_act,
                "month": 12 * 18,
                "fraction": 1.0,
            }
        )

        prev_fraction = 0.0
        for month in ages:
            speech_acts_children_month = data_children[data_children["age_months"] == month]
            children_ids = speech_acts_children_month["file_id"].unique()
            n_children = 0
            n_acquired = 0
            for child_id in children_ids:
                speech_acts_child = speech_acts_children_month[
                    speech_acts_children_month["file_id"] == child_id
                    ]
                if len(speech_acts_child) > MIN_NUM_UTTERANCES:
                    n_children += 1
                    target_speech_acts_child = speech_acts_child[
                        speech_acts_child[SPEECH_ACT] == speech_act
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

            fraction_acquired_speech_act.append(
                {
                    "speech_act": speech_act,
                    "month": month,
                    "fraction": fraction,
                }
            )
            prev_fraction = fraction

    return pd.DataFrame(fraction_acquired_speech_act)


if __name__ == "__main__":
    print("Loading data...")
    # TODO use classification scores on other dataset?
    scores = pickle.load(open("data/classification_scores_crf.p", "rb"))
    # scores = pickle.load(open("results/baseline/classification_scores_RF.p", "rb"))
    # scores = pickle.load(open("results/nn/classification_scores_lstm_baseline.p", "rb"))
    scores_f1 = scores["f1-score"].to_dict()

    # Calculate overall adult speech act frequencies
    data = pd.read_pickle('data/new_england_preprocessed.p')
    data_adults = data[data["speaker"] != "CHI"]
    data_children = data[data["speaker"] == "CHI"]

    frequencies_adults = calculate_frequencies(data_adults[SPEECH_ACT])
    frequencies_children = calculate_frequencies(data_children[SPEECH_ACT])
    frequencies = calculate_frequencies(data[SPEECH_ACT])

    # TODO
    # observed_speech_acts = [
    #     k for k, v in scores_f1.items() if v > 0.3 and k in frequencies_adults.keys()
    # ]
    observed_speech_acts = [k for k, v in frequencies.items() if k in scores_f1 and v > .01]
    # observed_speech_acts = [k for k, v in frequencies_children.items() if k in scores_f1]

    observed_speech_acts = [s for s in observed_speech_acts if s not in ["YY", "OO"]]

    ages = [14, 20, 32]
    # map ages to corresponding bins
    data_children["age_months"] = data_children["age_months"].apply(age_bin)

    # fraction_producing_speech_act = get_fraction_producing_speech_acts(data_children, ages, observed_speech_acts)
    fraction_contingent_responses = get_fraction_contingent_responses(ages, observed_speech_acts)
    fraction_data = fraction_contingent_responses

    # Filter data for observed speech acts
    frequencies_adults = [frequencies_adults[s] for s in observed_speech_acts]
    scores_f1 = [scores_f1[s] for s in observed_speech_acts]

    sns.set_palette(COLORS_PLOT_CATEGORICAL)
    g = sns.lmplot(
        data=fraction_data,
        x="month",
        y="fraction",
        hue="speech_act",
        logistic=True,
        ci=None,
        legend=True,
    )
    g.set(ylim=(0, 1), xlim=(min(ages)-4, max(ages)+12))
    plt.setp(g.legend.get_texts(), fontsize="10")

    # Read estimated ages of acquisition from the logistic regression plot data
    age_of_acquisition = {}
    for i, speech_act in enumerate(observed_speech_acts):
        fractions = g.ax.get_lines()[i].get_ydata()
        ages = g.ax.get_lines()[i].get_xdata()

        # If the logistic regression has failed: use data from points
        # TODO: improve.
        if np.isnan(fractions).all():
            warnings.warn(f"Couldn't calculate logistic regression for {speech_act}")
            fractions_speech_act_acquired = fraction_data[
                (fraction_data["speech_act"] == speech_act)
                & fraction_data["fraction"]
                >= THRESHOLD_FRACTION_ACQUIRED
                ]
            if len(fractions_speech_act_acquired) > 0:
                age_of_acquisition[speech_act] = min(
                    fraction_data["month"]
                )
            else:
                age_of_acquisition[speech_act] = max(ages)

        # Take data from logistic regression curve
        else:
            if np.where(fractions > 0.5)[0].size > 0:
                age_of_acquisition[speech_act] = ages[np.min(np.where(fractions >= THRESHOLD_FRACTION_ACQUIRED))]
            else:
                age_of_acquisition[speech_act] = max(ages)
        print(
            f"Age of acquisition of {speech_act}: {age_of_acquisition[speech_act]:.1f} |"
            f" Freq: {frequencies_adults[i]} | F1: {scores_f1[i]}"
        )

    plt.show(block=False)


    fig, ax = plt.subplots()
    x = list(age_of_acquisition.values())
    y = list(scores_f1)
    ax.scatter(x, y)
    plt.xlabel("age of acquisition (months)")
    plt.ylabel("classification score (F1)")

    for i, speech_act in enumerate(observed_speech_acts):
        ax.annotate(speech_act, (x[i], y[i]))

    fig, ax = plt.subplots()
    x = list(age_of_acquisition.values())
    y = list(frequencies_adults)
    ax.scatter(x, y)
    plt.xlabel("age of acquisition (months)")
    plt.ylabel("frequency (%)")

    for i, speech_act in enumerate(observed_speech_acts):
        ax.annotate(speech_act, (x[i], y[i]))

    plt.show(block=False)

    features = np.array(frequencies_adults).reshape(-1, 1)
    targets = list(age_of_acquisition.values())
    reg = LinearRegression().fit(features, targets)
    y_pred = reg.predict(features)
    print("Explained variance (only freq):", explained_variance_score(targets, y_pred))
    print("Regression parameters: ", reg.coef_)

    features = np.array(scores_f1).reshape(-1, 1)
    targets = list(age_of_acquisition.values())
    reg = LinearRegression().fit(features, targets)
    y_pred = reg.predict(features)
    print("Explained variance (only f1 scores):", explained_variance_score(targets, y_pred))
    print("Regression parameters: ", reg.coef_)

    features = np.array(list(zip(frequencies_adults, scores_f1)))
    reg = LinearRegression().fit(features, targets)
    y_pred = reg.predict(features)
    print(
        "Explained variance (freq + f1 scores):",
        explained_variance_score(targets, y_pred),
    )
    print("Regression parameters: ", reg.coef_)
    F, p_val = f_regression(features, targets)
    print("p-values: ", p_val)

    plt.show()
