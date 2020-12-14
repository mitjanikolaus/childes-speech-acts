import pickle
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.metrics import explained_variance_score

from crf_annotate import calculate_frequencies
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from preprocess import SPEECH_ACT
from utils import COLORS_PLOT_CATEGORICAL, age_bin

from statsmodels.api import Logit

MIN_NUM_UTTERANCES = 100
MIN_CHILDREN_REQUIRED = 3
THRESHOLD_ACQUIRED = 1
THRESHOLD_FRACTION_ACQUIRED = 0.5


if __name__ == "__main__":
    # Model prediction accuracies
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

    # ages = data_children["age_months"].unique()
    ages = [14, 20, 32]
    # map ages to corresponding bins
    data_children["age_months"] = data_children["age_months"].apply(age_bin)


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

    # for speech_act in observed_speech_acts:
    #     fractions_speech_act = fraction_acquired_speech_act[fraction_acquired_speech_act["speech_act"] == speech_act]
    #     months = fractions_speech_act["month"].values
    #     fractions = fractions_speech_act["fraction"].values
    #     log_reg = Logit(fractions, months).fit()
    #
    #     pred_input = np.linspace(min(ages)-4, max(ages)+12, 100)
    #     predictions = log_reg.predict(pred_input)
    #
    #     plt.scatter(months, fractions)
    #     plt.plot(pred_input, predictions)
    #     plt.xlim(min(ages)-4, max(ages)+12)
    #     plt.show()
    #     print("greoihg")


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
            fractions_speech_act_acquired = fraction_acquired_speech_act[
                (fraction_acquired_speech_act["speech_act"] == speech_act)
                & fraction_acquired_speech_act["fraction"]
                >= THRESHOLD_FRACTION_ACQUIRED
            ]
            if len(fractions_speech_act_acquired) > 0:
                age_of_acquisition[speech_act] = min(
                    fraction_acquired_speech_act["month"]
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
            f" Freq: {frequencies_adults_observed[i]} | F1: {scores_observed[i]}"
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
