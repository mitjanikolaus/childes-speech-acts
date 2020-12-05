import os
import pickle
import argparse
from collections import Counter
import json

import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import pycrfsuite

from crf_test import crf_predict
from crf_train import get_features_from_row
from preprocess import CHILD, ADULT


def parse_args():
    argparser = argparse.ArgumentParser(description="Annotate.")
    argparser.add_argument(
        "--model",
        "-m",
        required=True,
        type=str,
        default=None,
        help="folder containing model, features and metadata",
    )
    argparser.add_argument(
        "--data", type=str, required=True, help="Path to preprocessed data to annotate"
    )
    argparser.add_argument(
        "--compare", type=str, help="Path to frequencies to compare to"
    )

    args = argparser.parse_args()

    return args


def compare_frequencies(frequencies, args):
    gold_frequencies = pickle.load(open(args.compare, "rb"))
    frequencies = {k: frequencies[k] for k in gold_frequencies.keys()}
    kl_divergence = entropy(
        list(frequencies.values()), qk=list(gold_frequencies.values())
    )
    print(f"KL Divergence: {kl_divergence:.3f}")
    labels = list(gold_frequencies.keys()) * 2
    source = ["Gold"] * len(gold_frequencies) + ["Predicted"] * len(gold_frequencies)
    frequencies = list(gold_frequencies.values()) + list(frequencies.values())
    df = pd.DataFrame(zip(labels, source, frequencies), columns=["speech_act", "source", "frequency"])
    plt.figure(figsize=(10, 6))
    sns.barplot(x="speech_act", hue="source", y="frequency", data=df)
    plt.title(f"{args.data} compared to {args.compare} | KL Divergence: {kl_divergence:.3f}")
    plt.show()

def calculate_frequencies(data: list):
    frequencies = Counter(data)
    for k in frequencies.keys():
        if frequencies[k]:
            frequencies[k] /= len(data)
        else:
            frequencies[k] = 0

    return frequencies

if __name__ == "__main__":
    args = parse_args()

    # Loading data
    data = pickle.load(open(args.data, "rb"))

    # Loading model
    model_path = args.model + os.sep + "model.pycrfsuite"
    features_path = args.model + os.sep + "features.json"

    # TODO load metadata to know used features

    data = pd.DataFrame(data).reset_index(drop=False)

    # Add turn length column
    data["turn_length"] = data.tokens.apply(len)

    # Replace speaker column values
    data["speaker"] = data["speaker"].apply(
        lambda x: CHILD if x == "Target_Child" else ADULT
    )

    # Loading features
    with open(features_path, "r") as json_file:
        features_idx = json.load(json_file)

    # TODO use more features?
    data["features"] = data.apply(
        lambda x: get_features_from_row(
            features_idx,
            x.tokens,
            x["speaker"],
            x.turn_length,
            use_bi_grams=args.use_bi_grams,
        ),
        axis=1,
    )

    # Predictions
    tagger = pycrfsuite.Tagger()
    tagger.open(model_path)

    # creating data
    X_dev = data.groupby(by=["file_id"]).agg(
        {"features": lambda x: [y for y in x], "index": min}
    )
    y_pred = crf_predict(
        tagger,
        X_dev.sort_values("index", ascending=True)["features"],
    )
    data["y_pred"] = [y for x in y_pred for y in x]

    # Filter for children's and adults' utterances
    data_children = data[data.speaker == CHILD]
    data_adults = data[data.speaker != CHILD]

    speech_acts_adults = data_adults["y_pred"].tolist()

    pickle.dump(data_children[["file_id", "y_pred"]], open(args.data.replace("utterances", "speech_acts_chi"), "wb"))
    pickle.dump(data_adults[["file_id", "y_pred"]], open(args.data.replace("utterances", "speech_acts_adu"), "wb"))

    frequencies_children = calculate_frequencies(data_children["y_pred"].tolist())

    if args.compare:
        compare_frequencies(frequencies_children, args)

    #
    # for _, row in data.iterrows():
    #     print(f"({row.y_pred}) {row.speaker}: {' '.join(row.tokens)}")

