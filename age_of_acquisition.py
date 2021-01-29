import argparse
import pickle
import warnings

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from preprocess import SPEECH_ACT, CHILD
from utils import COLORS_PLOT_CATEGORICAL, age_bin, COLLAPSED_FORCE_CODES_TRANSLATIONS

MIN_NUM_UTTERANCES = 0
MIN_CHILDREN_REQUIRED = 0
THRESHOLD_ACQUIRED = 2
THRESHOLD_FRACTION_ACQUIRED = 0.5

THRESHOLD_SPEECH_ACT_OBSERVED_PRODUCTION = 0

MIN_AGE = 6
MAX_AGE = 12 * 18

ADD_EXTRA_DATAPOINTS = False

COMPREHENSION_DATA_POINTS_2_OCCURRENCES = {14: ['STAPDWTXCT', 'MKTOXA', 'SA', 'AC', 'RPCS', 'CRDS', 'YQTQ', 'RQ', 'CMEMED', 'YYOO', 'QN', 'PRCN', 'RR', 'EAEIECEX', 'YA', 'SS', 'DC', 'PF', 'CL', 'ABPMET', 'SI', 'EQ', 'AA', 'RT', 'GR', 'PA', 'FP', 'ADALGI', 'WD', 'RD', 'PD'], 20: ['QN', 'RPCS', 'PD', 'ABPMET', 'SS', 'MKTOXA', 'YYOO', 'STAPDWTXCT', 'AC', 'AA', 'RQ', 'RT', 'SI', 'PRCN', 'PF', 'CL', 'ADALGI', 'YQTQ', 'RD', 'YA', 'EAEIECEX', 'DC', 'WD', 'SA', 'GR', 'EQ', 'RR', 'YD', 'CRDS', 'AN', 'CMEMED', 'FP', 'AQ', 'PA'], 32: ['QN', 'YQTQ', 'MKTOXA', 'STAPDWTXCT', 'SA', 'AC', 'RR', 'FP', 'RPCS', 'RQ', 'ADALGI', 'PRCN', 'ABPMET', 'SI', 'PA', 'YYOO', 'YA', 'AQ', 'CMEMED', 'CL', 'RT', 'GR', 'PF', 'EAEIECEX', 'AA', 'SS', 'EQ', 'DC', 'RD', 'AN', 'WD', 'CXSC', 'PD']}
COMPREHENSION_SPEECH_ACTS_ENOUGH_DATA_2_OCCURRENCES =['STAPDWTXCT', 'MKTOXA', 'SA', 'AC', 'RPCS', 'CRDS', 'YQTQ', 'RQ', 'CMEMED', 'QN', 'PRCN', 'RR', 'EAEIECEX', 'YA', 'SS', 'DC', 'PF', 'CL', 'ABPMET', 'SI', 'EQ', 'AA', 'RT', 'GR', 'PA', 'FP', 'ADALGI', 'WD', 'RD', 'PD', 'AN', 'AQ']

COMPREHENSION_DATA_POINTS = COMPREHENSION_DATA_POINTS_2_OCCURRENCES
COMPREHENSION_SPEECH_ACTS = COMPREHENSION_SPEECH_ACTS_ENOUGH_DATA_2_OCCURRENCES

def get_fraction_contingent_responses(ages, observed_speech_acts, add_extra_datapoints=False):
    """Calculate "understanding" of speech acts by measuring the amount of contingent responses"""
    fraction_contingent_responses = []

    for month in ages:
        contingency_data = pd.read_csv(
            f"adjacency_pairs/ADU-CHI_age_{month}_collapsed_contingency.csv"
        )

        for speech_act in observed_speech_acts:
            if add_extra_datapoints:
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

            if speech_act in COMPREHENSION_DATA_POINTS[month]:
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


def get_fraction_producing_speech_acts(data_children, ages, observed_speech_acts, column_name_speech_act=SPEECH_ACT,
                                       add_extra_datapoints=True):
    fraction_acquired_speech_act = []

    print("Processing speech acts...")
    for speech_act in observed_speech_acts:

        if add_extra_datapoints:
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
                        speech_acts_child[column_name_speech_act] == speech_act
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

def calc_ages_of_acquisition(target, data, observed_speech_acts, ages, column_name_speech_act=SPEECH_ACT, add_extra_datapoints=True, max_age=MAX_AGE):

    if target == TARGET_PRODUCTION:
        data_children = data[data["speaker"] == CHILD]

        observed_speech_acts = [
            s
            for s in observed_speech_acts
            if s in data_children[column_name_speech_act].unique()
               and data_children[column_name_speech_act].value_counts()[s]
               > THRESHOLD_SPEECH_ACT_OBSERVED_PRODUCTION
        ]

        fraction_producing_speech_act = get_fraction_producing_speech_acts(
            data_children, ages, observed_speech_acts, column_name_speech_act, add_extra_datapoints
        )

        fraction_data = fraction_producing_speech_act

    elif target == TARGET_COMPREHENSION:
        observed_speech_acts = COMPREHENSION_SPEECH_ACTS

        fraction_contingent_responses = get_fraction_contingent_responses(
            ages, observed_speech_acts, add_extra_datapoints
        )

        fraction_data = fraction_contingent_responses

    sns.set_palette(COLORS_PLOT_CATEGORICAL)

    g = sns.FacetGrid(data=fraction_data, hue="speech_act")
    g.set(ylim=(0, 1), xlim=(min(ages) - 4, max_age))
    g.map(sns.regplot, "month", "fraction", truncate=False, logistic=True, ci=None)
    h, l = g.axes[0][0].get_legend_handles_labels()
    g.fig.legend(h, l, loc='upper center', ncol=8)
    plt.subplots_adjust(top=0.71, bottom=0.09, left=0.057)
    plt.xlabel("age (months)")
    if target == TARGET_PRODUCTION:
        plt.ylabel("fraction of children producing the target speech act")
    elif target == TARGET_COMPREHENSION:
        plt.ylabel("fraction of contingent responses")

    # Read estimated ages of acquisition from the logistic regression plot data
    age_of_acquisition = {}
    for i, speech_act in enumerate(observed_speech_acts):
        fractions = g.ax.get_lines()[i].get_ydata()
        ages = g.ax.get_lines()[i].get_xdata()

        # If the logistic regression has failed: use data from points
        if np.isnan(fractions).all():
            warnings.warn(f"Couldn't calculate logistic regression for {speech_act}. Setting AoA to max_age.")
            age_of_acquisition[speech_act] = max_age

        # Take data from logistic regression curve
        else:
            if np.where(fractions >= THRESHOLD_FRACTION_ACQUIRED)[0].size > 0:
                age_of_acquisition[speech_act] = ages[
                    np.min(np.where(fractions >= THRESHOLD_FRACTION_ACQUIRED))
                ]
            else:
                age_of_acquisition[speech_act] = max_age
        print(
            f"Age of acquisition of {speech_act}: {age_of_acquisition[speech_act]:.1f}"
        )


    return age_of_acquisition


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

    data = pd.read_pickle("data/new_england_preprocessed.p")

    data[SPEECH_ACT] = data[SPEECH_ACT].apply(
        lambda x: COLLAPSED_FORCE_CODES_TRANSLATIONS.loc[x].Group
    )

    # map ages to corresponding bins
    data["age_months"] = data["age_months"].apply(age_bin)

    observed_speech_acts = [label for label in data[SPEECH_ACT].unique()]

    # Filter out unintelligible acts
    observed_speech_acts = [s for s in observed_speech_acts if s not in ["YYOO"]]

    ages = [14, 20, 32]

    ages_of_acquisition = calc_ages_of_acquisition(args.target, data, observed_speech_acts, ages, add_extra_datapoints=ADD_EXTRA_DATAPOINTS)

    path = f"results/age_of_acquisition_{args.target}_collapsed.p"
    pickle.dump(ages_of_acquisition, open(path, "wb"))
    ages_of_acquisition = pd.DataFrame(ages_of_acquisition, index=[0])
    ages_of_acquisition.to_csv(path.replace(".p", ".csv"))

    plt.show()
