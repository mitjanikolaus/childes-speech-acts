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

from crf_test import crf_predict, load_training_args
from crf_train import get_features_from_row, add_feature_columns
from preprocess import CHILD, ADULT
from utils import calculate_frequencies


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
    df = pd.DataFrame(
        zip(labels, source, frequencies), columns=["speech_act", "source", "frequency"]
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(x="speech_act", hue="source", y="frequency", data=df)
    plt.title(
        f"{args.data} compared to {args.compare} | KL Divergence: {kl_divergence:.3f}"
    )
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    args = load_training_args(args)
    print(args)

    # Loading data
    data = pd.read_hdf(args.data)

    # Loading model
    model_path = args.model + os.sep + "model.pycrfsuite"
    features_path = args.model + os.sep + "feature_vocabs.p"

    data = data.reset_index(drop=False)

    # Replace speaker column values
    data["speaker"] = data["speaker"].apply(
        lambda x: CHILD if x == "Target_Child" else ADULT
    )

    # Loading features
    with open(features_path, "rb") as pickle_file:
        feature_vocabs = pickle.load(pickle_file)

    data = add_feature_columns(
        data,
        use_action=args.use_action,
        match_age=args.match_age,
        check_repetition=args.use_repetitions,
        use_past=args.use_past,
        use_pastact=args.use_past_actions,
        use_pos=args.use_pos,
    )

    data["features"] = data.apply(
        lambda x: get_features_from_row(
            feature_vocabs,
            x.tokens,
            x["speaker"],
            x["prev_speaker"],
            x.turn_length,
            use_bi_grams=args.use_bi_grams,
            action_tokens=None if not args.use_action else x.action_tokens,
            repetitions=None
            if not args.use_repetitions
            else (x.repeated_words, x.nb_repwords, x.ratio_repwords),
            past_tokens=None if not args.use_past else x.past,
            pastact_tokens=None if not args.use_past_actions else x.past_act,
            pos_tags=None if not args.use_pos else x.pos,
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

    data_children.to_hdf("data/speech_acts_chi.h5", key="speech_acts")
    data_adults.to_hdf("data/speech_acts_adu.h5", key="speech_acts")

    if args.compare:
        frequencies_children = calculate_frequencies(data_children["y_pred"].tolist())
        compare_frequencies(frequencies_children, args)

    #
    # for _, row in data.iterrows():
    #     print(f"({row.y_pred}) {row.speaker}: {' '.join(row.tokens)}")
