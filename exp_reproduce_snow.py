import pickle

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from preprocess import SPEECH_ACT, CHILD
from utils import age_bin, calculate_frequencies

AGES = [14, 20, 32]

def calculate_num_speech_act_types(data, column_name_speech_act):
    # number of speech act types at different ages
    results = []
    for age in AGES:
        data_age = data[data["age_months"] == age]

        results_age = {}
        for num_speech_act_types in range(20):
            results_age[num_speech_act_types] = 0

        children_ids = data_age.file_id.unique()
        for child_id in children_ids:
            speech_acts_child = data_age[
                (data_age.file_id == child_id)
                & (data_age.speaker == CHILD)
                & (~data_age[column_name_speech_act].isin(["YY", "OO"]))
            ][column_name_speech_act]
            speech_act_type_counts = speech_acts_child.value_counts().to_list()

            num_produced_speech_act_types = len(
                [c for c in speech_act_type_counts if c >= 2]
            )

            results_age[num_produced_speech_act_types] += 1

        for num_speech_act_types in range(20):
            results.append(
                {
                    "age": age,
                    "num_types": num_speech_act_types,
                    "num_children": results_age[num_speech_act_types],
                }
            )

    return pd.DataFrame(results)

def reproduce_num_speech_acts(data):
    results_snow = calculate_num_speech_act_types(data, SPEECH_ACT)
    results_crf = calculate_num_speech_act_types(data, "y_pred")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex="all", sharey="all")
    sns.barplot(ax=ax1, x="num_types", hue="age", y="num_children", data=results_snow)

    # Move title into figure
    plt.rcParams['axes.titley'] = 1.0
    plt.rcParams['axes.titlepad'] = -14

    ax1.set_title("Data from Snow et. al. (1996)")

    ax1.set_xlabel("")
    ax1.set_ylabel("number of children")

    sns.barplot(ax=ax2, x="num_types", hue="age", y="num_children", data=results_crf)

    # Move title into figure
    plt.rcParams['axes.titley'] = 1.0
    plt.rcParams['axes.titlepad'] = -14

    ax2.set_title("Automatically Annotated Data")
    ax2.legend_.remove()
    ax2.set_ylabel("number of children")
    plt.xlabel("number of different speech acts produced")
    plt.tight_layout()
    plt.show()

def calculate_freq_distributions(data, column_name_speech_act, speech_acts_analyzed, age, source):
    # map ages to corresponding bins
    data["age_months"] = data["age_months"].apply(age_bin)

    # number of speech act types at different ages
    data_age = data[data["age_months"] == age]

    speech_acts_children = data_age[data_age.speaker == CHILD]

    frequencies = calculate_frequencies(speech_acts_children[column_name_speech_act])

    # Filter for speech acts analyzed
    frequencies = {s: f for s, f in frequencies.items() if s in speech_acts_analyzed}

    results = []
    for s, f in frequencies.items():
        results.append({
            "source": source,
            "speech_act": s,
            "frequency": f,
        })

    return results


def reproduce_speech_act_distribution(data):
    speech_acts_analyzed = ["YY", "ST", "PR", "MK", "SA", "RT", "RP", "RD", "AA", "AD", "AC", "QN", "YQ", "CL", "SI"]

    fig, axes = plt.subplots(3, 1, sharex="all") # sharey="all"

    for i, age in enumerate(AGES):
        results_snow = calculate_freq_distributions(data, SPEECH_ACT, speech_acts_analyzed, age, "Data from Snow et. al. (1996)")
        results_crf = calculate_freq_distributions(data, "y_pred", speech_acts_analyzed, age, "Automatically Annotated Data")

        results = pd.DataFrame(results_snow + results_crf)
        sns.barplot(ax=axes[i], x="speech_act", hue="source", y="frequency", data=results)

        # Move title into figure
        plt.rcParams['axes.titley'] = 1.0
        plt.rcParams['axes.titlepad'] = -14

        axes[i].set_title(f"Age: {age} months")

        axes[i].set_xlabel("")
        axes[i].set_ylabel("Frequency")
        axes[i].legend_.remove()

        if age > 14:
            axes[i].set_ylim(0, 0.3)

        # kl_divergence = entropy(
        #     list(counters["pred"].values()), qk=list(counters["gold"].values())
        # )
        # print(f"KL Divergence: {kl_divergence:.3f}")

    axes[-1].set_xlabel("Speech Act")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Model prediction accuracies
    print("Loading data...")
    data = pickle.load(open("data/new_england_reproduced_crf.p", "rb"))

    # map ages to corresponding bins
    data["age_months"] = data["age_months"].apply(age_bin)

    reproduce_speech_act_distribution(data)

    reproduce_num_speech_acts(data)

