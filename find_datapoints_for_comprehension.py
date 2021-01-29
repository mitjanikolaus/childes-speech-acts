import os
from collections import Counter

import pandas as pd
import matplotlib
from sklearn.preprocessing import OrdinalEncoder
import random

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

from preprocess import SPEECH_ACT, CHILD, ADULT
from utils import COLLAPSED_FORCE_CODES_TRANSLATIONS, COLLAPSED_FORCE_CODES, age_bin


def find_speech_acts(source=ADULT, target=CHILD, min_occurrences=0):
    # Load data
    data = pd.read_pickle("data/new_england_preprocessed.p")
    data[SPEECH_ACT] = data[SPEECH_ACT].apply(lambda x: COLLAPSED_FORCE_CODES_TRANSLATIONS.loc[x].Group)

    match_age = [14, 20, 32]
    # data["age_months"] = data["age_months"].apply(age_bin)
    data["age_months"] = data.age_months.apply(
        lambda age: min(match_age, key=lambda x: abs(x - age))
    )

    # Filter data
    speech_acts_datapoints = {}

    for age in match_age:
        data_age = data[data["age_months"] == age]
        # 1. Sequence extraction & columns names
        spa_shifted = {0: data_age[[SPEECH_ACT, "speaker", "file_id"]]}
        spa_shifted[1] = (
            spa_shifted[0]
                .shift(periods=1, fill_value=None)
                .rename(columns={col: col + "_1" for col in spa_shifted[0].columns})
        )
        spa_shifted[0] = spa_shifted[0].rename(
            columns={col: col + "_0" for col in spa_shifted[0].columns}
        )
        # 2. Merge
        spa_compare = pd.concat(spa_shifted.values(), axis=1)
        # 3. Add empty slots for file changes
        spa_compare.loc[
            (spa_compare["file_id_0"] != spa_compare["file_id_1"]), [f"{SPEECH_ACT}_1"]
        ] = None

        spa_sequences = spa_compare[[col for col in spa_compare.columns if "file_id" not in col]]

        # for now source = 1 and target = 0
        spa_sequences.rename({"speaker_1": "source", "speaker_0": "target"}, axis='columns', inplace=True)
        speaker_source = "source"
        speaker_target = "target"
        spa_source = "speech_act_1"
        spa_target = "speech_act_0"

        # 1. Choose illocutionary or interchange, remove unused sequences, remove NAs, select direction (MOT => CHI or CHI => MOT)
        spa_sequences.dropna(how="any", inplace=True)
        if source is not None and source in [CHILD, ADULT]:
            spa_sequences = spa_sequences[(spa_sequences[speaker_target] == target)]
        if target is not None and target in [CHILD, ADULT]:
            spa_sequences = spa_sequences[(spa_sequences[speaker_source] == source)]

        speech_acts_enough_data_age = []
        for speech_act_source in spa_sequences[spa_source].unique():
            num = len(spa_sequences[spa_sequences[spa_source] == speech_act_source])
            if num >= min_occurrences:
                speech_acts_enough_data_age.append(speech_act_source)

        speech_acts_datapoints[age] = speech_acts_enough_data_age

    print(speech_acts_datapoints)
    for age in match_age:
        print(speech_acts_datapoints[age])
        print(len(speech_acts_datapoints[age]))

    counter = Counter()
    for age in match_age:
        counter.update(speech_acts_datapoints[age])

    print("Enough datapoints: ", [s for s, o in counter.items() if o >= min_occurrences])


if __name__ == "__main__":
    find_speech_acts(min_occurrences=2)
