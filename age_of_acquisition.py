import argparse
import warnings

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from utils import SPEECH_ACT, CHILD, PATH_NEW_ENGLAND_UTTERANCES, AGES
from process_contingencies import get_contingency_data
from utils import COLORS_PLOT_CATEGORICAL, age_bin, SOURCE_SNOW, SOURCE_CRF, TARGET_PRODUCTION, TARGET_COMPREHENSION

MIN_NUM_UTTERANCES = 0
MIN_CHILDREN_REQUIRED = 0
THRESHOLD_ACQUIRED = 2
THRESHOLD_FRACTION_ACQUIRED = 0.5

THRESHOLD_SPEECH_ACT_OBSERVED_PRODUCTION = 0

MIN_AGE = 6
MAX_AGE = 12 * 18

ADD_EXTRA_DATAPOINTS = False

COMPREHENSION_DATA_POINTS_2_OCCURRENCES = {
    14: [
        "ST",
        "TO",
        "SA",
        "AC",
        "RP",
        "CR",
        "MK",
        "YQ",
        "RQ",
        "CM",
        "OO",
        "QN",
        "TX",
        "PR",
        "RR",
        "EA",
        "XA",
        "CS",
        "YA",
        "SS",
        "DC",
        "PF",
        "AP",
        "CL",
        "AB",
        "TQ",
        "SI",
        "EI",
        "EQ",
        "YY",
        "CT",
        "PM",
        "AA",
        "DW",
        "RT",
        "ET",
        "GR",
        "DS",
        "PA",
        "FP",
        "AD",
        "WD",
        "RD",
        "CN",
        "GI",
        "ED",
        "PD",
    ],
    20: [
        "QN",
        "RP",
        "PD",
        "PM",
        "SS",
        "MK",
        "OO",
        "AP",
        "ST",
        "AC",
        "AA",
        "RQ",
        "RT",
        "ET",
        "SI",
        "PR",
        "PF",
        "CL",
        "AD",
        "YQ",
        "AB",
        "RD",
        "YA",
        "EA",
        "DC",
        "GI",
        "WD",
        "SA",
        "YY",
        "GR",
        "EQ",
        "RR",
        "TO",
        "CS",
        "AL",
        "YD",
        "CT",
        "EI",
        "DS",
        "CR",
        "DW",
        "AN",
        "CM",
        "FP",
        "TQ",
        "TX",
        "XA",
        "CN",
        "AQ",
        "PA",
        "EC",
    ],
    32: [
        "QN",
        "YQ",
        "MK",
        "AP",
        "SA",
        "DW",
        "AC",
        "RR",
        "FP",
        "ST",
        "RP",
        "RQ",
        "AD",
        "PR",
        "PM",
        "XA",
        "SI",
        "GI",
        "PA",
        "OO",
        "TQ",
        "YA",
        "CS",
        "AQ",
        "TX",
        "CT",
        "CM",
        "CL",
        "RT",
        "GR",
        "PF",
        "EI",
        "AA",
        "SS",
        "EQ",
        "DC",
        "YY",
        "RD",
        "AB",
        "AN",
        "TO",
        "EC",
        "ET",
        "EX",
        "WD",
        "SC",
        "PD",
        "EA",
    ],
}
COMPREHENSION_SPEECH_ACTS_ENOUGH_DATA_2_OCCURRENCES = [
    "ST",
    "TO",
    "SA",
    "AC",
    "RP",
    "CR",
    "MK",
    "YQ",
    "RQ",
    "CM",
    "QN",
    "TX",
    "PR",
    "RR",
    "EA",
    "XA",
    "CS",
    "YA",
    "SS",
    "DC",
    "PF",
    "AP",
    "CL",
    "AB",
    "TQ",
    "SI",
    "EI",
    "EQ",
    "CT",
    "PM",
    "AA",
    "DW",
    "RT",
    "ET",
    "GR",
    "DS",
    "PA",
    "FP",
    "AD",
    "WD",
    "RD",
    "CN",
    "GI",
    "PD",
    "AN",
    "AQ",
    "EC",
]

COMPREHENSION_DATA_POINTS = COMPREHENSION_DATA_POINTS_2_OCCURRENCES
COMPREHENSION_SPEECH_ACTS = COMPREHENSION_SPEECH_ACTS_ENOUGH_DATA_2_OCCURRENCES


def get_fraction_contingent_responses(
    data, ages, observed_speech_acts, add_extra_datapoints=True, data_source=SOURCE_SNOW,
):
    """Calculate "understanding" of speech acts by measuring the amount of contingent responses"""
    if data_source not in (SOURCE_SNOW, SOURCE_CRF):
        raise ValueError("Unknown data source: ", data_source)

    fraction_contingent_responses = []

    for age_months in ages:
        contingency_data = get_contingency_data(data, age_months, data_source)

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

            if speech_act in COMPREHENSION_DATA_POINTS[age_months]:
                fraction = contingency_data[
                    (contingency_data["source"] == speech_act)
                    & (contingency_data["contingency"] == 1)
                ]["fraction"].sum()

                fraction_contingent_responses.append(
                    {
                        "speech_act": speech_act,
                        "month": age_months,
                        "fraction": fraction,
                    }
                )

    return pd.DataFrame(fraction_contingent_responses)


def get_fraction_producing_speech_acts(
    data_children,
    ages,
    observed_speech_acts,
    column_name_speech_act=SPEECH_ACT,
    add_extra_datapoints=True,
):
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
                data_children.age == month
            ]
            transcript_ids = speech_acts_children_month.transcript_file.unique()
            n_children = 0
            n_acquired = 0
            for transcript_id in transcript_ids:
                speech_acts_child = speech_acts_children_month[
                    speech_acts_children_month.transcript_file == transcript_id
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


def calc_ages_of_acquisition(
    target,
    data,
    observed_speech_acts,
    ages,
    data_source=SOURCE_SNOW,
    add_extra_datapoints=True,
    max_age=MAX_AGE,
    threshold_speech_act_observed_production=THRESHOLD_SPEECH_ACT_OBSERVED_PRODUCTION,
):

    if target == TARGET_PRODUCTION:
        data_children = data[data.speaker_code == CHILD]

        if data_source == SOURCE_SNOW:
            column_name_speech_act = SPEECH_ACT
        elif data_source == SOURCE_CRF:
            column_name_speech_act = "y_pred"
        else:
            raise ValueError("Unknown data source: ", data_source)

        observed_speech_acts = [
            s
            for s in observed_speech_acts
            if s in data_children[column_name_speech_act].unique()
            and data_children[column_name_speech_act].value_counts()[s]
            > threshold_speech_act_observed_production
        ]
        # observed_speech_acts = ["ST","MK","YQ","SA","RR","FP"]

        fraction_producing_speech_act = get_fraction_producing_speech_acts(
            data_children,
            ages,
            observed_speech_acts,
            column_name_speech_act,
            add_extra_datapoints,
        )

        fraction_data = fraction_producing_speech_act

    elif target == TARGET_COMPREHENSION:
        fraction_contingent_responses = get_fraction_contingent_responses(
            data, ages, observed_speech_acts, add_extra_datapoints, data_source
        )

        fraction_data = fraction_contingent_responses

    sns.set_palette(COLORS_PLOT_CATEGORICAL)

    g = sns.FacetGrid(data=fraction_data, hue="speech_act", height=6, aspect=1.3)
    g.set(ylim=(0, 1), xlim=(min(ages) - 4, max_age))
    g.map(sns.regplot, "month", "fraction", truncate=False, logistic=True, ci=None)

    h, l = g.axes[0][0].get_legend_handles_labels()
    g.fig.legend(h, l, loc="upper center", ncol=10)
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
            warnings.warn(
                f"Couldn't calculate logistic regression for {speech_act}. Setting AoA to max_age."
            )
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
        
    # Reset color palette to default for upcoming plots
    sns.set_palette("tab10")

    return age_of_acquisition


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--target",
        type=str,
        default=TARGET_PRODUCTION,
        choices=[TARGET_PRODUCTION, TARGET_COMPREHENSION],
    )
    argparser.add_argument(
        "--data-source",
        type=str,
        default=SOURCE_SNOW,
        choices=[SOURCE_SNOW, SOURCE_CRF],
    )

    args = argparser.parse_args()

    print("Loading data...")

    data = pd.read_pickle(PATH_NEW_ENGLAND_UTTERANCES)

    # map ages to corresponding bins
    data["age"] = data["age"].apply(age_bin)

    observed_speech_acts = [label for label in data[SPEECH_ACT].unique()]

    # Filter out unintelligible acts
    observed_speech_acts = [s for s in observed_speech_acts if s not in ["YY", "OO"]]

    if args.target == TARGET_COMPREHENSION:
        observed_speech_acts = COMPREHENSION_SPEECH_ACTS

    ages_of_acquisition = calc_ages_of_acquisition(
        args.target,
        data,
        observed_speech_acts,
        AGES,
        data_source=args.data_source,
        add_extra_datapoints=ADD_EXTRA_DATAPOINTS,
    )

    path = f"results/age_of_acquisition_{args.target}_{args.data_source}.csv"
    ages_of_acquisition = pd.DataFrame.from_records([ages_of_acquisition]).T
    ages_of_acquisition.index.rename('speech_act', inplace=True)
    ages_of_acquisition.rename(columns={0: "age_of_acquisition"}, inplace=True)
    ages_of_acquisition.to_csv(path)

    plt.axhline(y=0.5, linestyle="--")
    plt.xlim(10, 60)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.09)
    plt.show()
