import pickle

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from preprocess import SPEECH_ACT, CHILD
from utils import age_bin


def calculate_num_speech_act_types(data, column_name_speech_act):
    ages = [14, 20, 32]
    # map ages to corresponding bins
    data["age_months"] = data["age_months"].apply(age_bin)

    # number of speech act types at different ages
    results = []
    for age in ages:
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



if __name__ == "__main__":
    # Model prediction accuracies
    print("Loading data...")
    data = pickle.load(open("data/new_england_reproduced_crf.p", "rb"))


    results_snow = calculate_num_speech_act_types(data, SPEECH_ACT)
    results_crf = calculate_num_speech_act_types(data, "y_pred")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex="all", sharey="all")
    sns.barplot(ax=ax1, x="num_types", hue="age", y="num_children", data=results_snow)
    ax1.set_title("Data from Snow et. al. (1996)")
    ax1.set_xlabel("")
    ax1.set_ylabel("number of children")

    sns.barplot(ax=ax2, x="num_types", hue="age", y="num_children", data=results_crf)
    ax2.set_title("Automatically Annotated Data")
    ax2.legend_.remove()
    ax2.set_ylabel("number of children")
    plt.xlabel("number of different speech acts produced")
    plt.tight_layout()
    plt.show()

