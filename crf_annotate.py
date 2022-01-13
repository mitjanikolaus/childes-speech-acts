import os
import pickle
import argparse
from pathlib import Path

import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import pycrfsuite

from crf_test import crf_predict
from crf_train import get_features_from_row, add_feature_columns
from utils import CHILD
from utils import calculate_frequencies


def parse_args():
    argparser = argparse.ArgumentParser(description="Annotate.")
    argparser.add_argument(
        "--model",
        "-m",
        required=True,
        type=str,
        help="folder containing model and features",
    )
    argparser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to CSV with preprocessed data to annotate",
    )
    argparser.add_argument(
        "--out", type=str, required=True, help="Directory to store output files."
    )
    argparser.add_argument(
        "--compare", type=str, help="Path to frequencies to compare to"
    )
    argparser.add_argument(
        "--use-bi-grams",
        "-bi",
        action="store_true",
        help="whether to use bi-gram features to train the algorithm",
    )
    argparser.add_argument(
        "--use-pos",
        "-pos",
        action="store_true",
        help="whether to add POS tags to features",
    )
    argparser.add_argument(
        "--use-past",
        "-past",
        action="store_true",
        help="whether to add previous sentence as features",
    )
    argparser.add_argument(
        "--use-repetitions",
        "-rep",
        action="store_true",
        help="whether to check in data if words were repeated from previous sentence, to train the algorithm",
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
    print(args)

    # Loading data
    data = pd.read_pickle(args.data)

    # Loading model
    model_path = os.path.join(args.model, "model.pycrfsuite")
    features_path = os.path.join(args.model, "feature_vocabs.p")

    # Loading features
    with open(features_path, "rb") as pickle_file:
        feature_vocabs = pickle.load(pickle_file)

    data = add_feature_columns(
        data, check_repetition=args.use_repetitions, use_past=args.use_past,
    )

    data = data.assign(
        features=data.apply(
            lambda x: get_features_from_row(
                feature_vocabs,
                x.tokens,
                x.speaker_code,
                x.prev_speaker_code,
                x.turn_length,
                use_bi_grams=args.use_bi_grams,
                repetitions=None
                if not args.use_repetitions
                else (x.repeated_words, x.ratio_repwords),
                prev_tokens=None if not args.use_past else x.past,
                pos_tags=None if not args.use_pos else x.pos,
            ),
            axis=1,
        )
    )

    # Predictions
    tagger = pycrfsuite.Tagger()
    tagger.open(model_path)

    # creating data
    X_dev = data.groupby(by=["transcript_file"]).agg(
        {"features": lambda x: [y for y in x]}
    )
    y_pred = crf_predict(tagger, X_dev["features"],)
    data = data.assign(speech_act=[y for x in y_pred for y in x])  # flatten

    # Filter for important columns
    data_filtered = data.drop(
        columns=[
            "prev_tokens",
            "prev_speaker_code",
            "repeated_words",
            "nb_repwords",
            "ratio_repwords",
            "features",
        ]
    )

    Path(args.out).mkdir(parents=True, exist_ok=True)
    data_filtered.to_pickle(
        os.path.join(args.out, "utterances_annotated_with_speech_acts.p")
    )

    if args.compare:
        data_children = data_filtered[data.speaker_code == CHILD]
        frequencies_children = calculate_frequencies(data_children["y_pred"].tolist())
        compare_frequencies(frequencies_children, args)
