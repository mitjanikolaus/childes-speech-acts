import argparse
import pickle
import warnings

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

THRESHOLD_SPEECH_ACT_OBSERVED = 0

MIN_AGE = 6
MAX_AGE = 12 * 18


def get_fraction_contingent_responses(ages, observed_speech_acts):
    """Calculate "understanding" of speech acts by measuring the amount of contingent responses"""
    fraction_contingent_responses = []

    for month in ages:
        contingency_data = pd.read_csv(
            f"adjacency_pairs/ADU-CHI_age_{month}_contingency.csv"
        )

        for speech_act in observed_speech_acts:
            # Add start: at 6 months children don't produce any speech act
            fraction_contingent_responses.append(
                {
                    "speech_act": speech_act,
                    "month": MIN_AGE,
                    "fraction": 0.0,
                }
            )
            # Add end: at 18 years children know all speech acts
            fraction_contingent_responses.append(
                {
                    "speech_act": speech_act,
                    "month": MAX_AGE,
                    "fraction": 1.0,
                }
            )

            fraction = contingency_data[
                (contingency_data["source"] == speech_act)
                & (contingency_data["contingency"] == 1)
            ]["fraction"].sum()

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

        # Add start: at 6 months children don't produce any speech act
        fraction_acquired_speech_act.append(
            {
                "speech_act": speech_act,
                "month": MIN_AGE,
                "fraction": 0.0,
            }
        )
        # Add end: at 18 years children know all speech acts
        fraction_acquired_speech_act.append(
            {
                "speech_act": speech_act,
                "month": MAX_AGE,
                "fraction": 1.0,
            }
        )

        prev_fraction = 0.0
        for month in ages:
            speech_acts_children_month = data_children[
                data_children["age_months"] == month
            ]
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


TARGET_PRODUCTION = "production"
TARGET_COMPREHENSION = "comprehension"

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--target",
        type=str,
        default=TARGET_PRODUCTION,
        choices=[TARGET_PRODUCTION, TARGET_COMPREHENSION],
    )

    args = argparser.parse_args()

    print("Loading data...")
    # Calculate overall adult speech act frequencies
    data = pd.read_pickle("data/new_england_preprocessed.p")
    data_adults = data[data["speaker"] != "CHI"]
    data_children = data[data["speaker"] == "CHI"]

    observed_speech_acts = [label for label in data[SPEECH_ACT].unique()]

    # Filter out unintelligible acts
    observed_speech_acts = [s for s in observed_speech_acts if s not in ["YY", "OO"]]

    ages = [14, 20, 32]
    # map ages to corresponding bins
    data_children["age_months"] = data_children["age_months"].apply(age_bin)

    if args.target == TARGET_PRODUCTION:
        observed_speech_acts = [
            s
            for s in observed_speech_acts
            if s in data_children[SPEECH_ACT].unique()
            and data_children[SPEECH_ACT].value_counts()[s]
            > THRESHOLD_SPEECH_ACT_OBSERVED
        ]

        fraction_producing_speech_act = get_fraction_producing_speech_acts(
            data_children, ages, observed_speech_acts
        )

        fraction_data = fraction_producing_speech_act

    elif args.target == TARGET_COMPREHENSION:
        observed_speech_acts = [
            s
            for s in observed_speech_acts
            if s in data_adults[SPEECH_ACT].unique()
            and data_adults[SPEECH_ACT].value_counts()[s]
            > THRESHOLD_SPEECH_ACT_OBSERVED
        ]

        fraction_contingent_responses = get_fraction_contingent_responses(
            ages, observed_speech_acts
        )

        fraction_data = fraction_contingent_responses

    sns.set_palette(COLORS_PLOT_CATEGORICAL)
    g = sns.lmplot(
        data=fraction_data,
        x="month",
        y="fraction",
        hue="speech_act",
        logistic=True,
        ci=None,
        legend_out=True,
        legend=False,
    )
    g.set(ylim=(0, 1), xlim=(min(ages) - 4, max(ages) + 12))
    h, l = g.axes[0][0].get_legend_handles_labels()
    g.fig.legend(h, l, loc='upper center', ncol=10)
    plt.subplots_adjust(top=0.7, bottom=0.09, left=0.057)
    plt.xlabel("age (months)")
    if args.target == TARGET_PRODUCTION:
        plt.ylabel("fraction of children producing the target speech act")
    elif args.target == TARGET_COMPREHENSION:
        plt.ylabel("fraction of contingent responses")

    # Read estimated ages of acquisition from the logistic regression plot data
    age_of_acquisition = {}
    for i, speech_act in enumerate(observed_speech_acts):
        fractions = g.ax.get_lines()[i].get_ydata()
        ages = g.ax.get_lines()[i].get_xdata()

        # If the logistic regression has failed: use data from points
        if np.isnan(fractions).all():
            warnings.warn(f"Couldn't calculate logistic regression for {speech_act}")
            fractions_speech_act_acquired = fraction_data[
                (fraction_data["speech_act"] == speech_act)
                & (fraction_data["fraction"] >= THRESHOLD_FRACTION_ACQUIRED)
            ]
            if len(fractions_speech_act_acquired) > 0:
                age_of_acquisition[speech_act] = min(
                    fractions_speech_act_acquired["month"]
                )
            else:
                age_of_acquisition[speech_act] = MAX_AGE

        # Take data from logistic regression curve
        else:
            if np.where(fractions >= THRESHOLD_FRACTION_ACQUIRED)[0].size > 0:
                age_of_acquisition[speech_act] = ages[
                    np.min(np.where(fractions >= THRESHOLD_FRACTION_ACQUIRED))
                ]
            else:
                age_of_acquisition[speech_act] = MAX_AGE
        print(
            f"Age of acquisition of {speech_act}: {age_of_acquisition[speech_act]:.1f} "
        )

    path = f"results/age_of_acquisition_{args.target}.p"
    pickle.dump(age_of_acquisition, open(path, "wb"))

    plt.show()
