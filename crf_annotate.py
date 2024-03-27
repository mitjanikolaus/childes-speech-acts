import os
import pickle
import argparse
from ast import literal_eval

import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
import pycrfsuite

from crf_test import crf_predict, update_args
from crf_train import get_features_from_row, add_feature_columns
from utils import CHILD, SPEECH_ACT_DESCRIPTIONS
from utils import calculate_frequencies


def parse_args():
    argparser = argparse.ArgumentParser(description="Annotate.")
    argparser.add_argument(
        "--checkpoint",
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
        "--out", type=str, required=True, help="Dir to store output files."
    )
    argparser.add_argument(
        "--compare", type=str, help="Path to frequencies to compare to"
    )
    argparser.add_argument(
        "--output-for-childes-db", action="store_true", default=False,
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

    if args.data.endswith(".csv"):
        data = pd.read_csv(args.data, converters={"pos": literal_eval, "tokens": literal_eval})
    else:
        data = pd.read_pickle(args.data)

    # Loading model
    model_path = os.path.join(args.checkpoint, "model.pycrfsuite")
    features_path = os.path.join(args.checkpoint, "feature_vocabs.p")
    args_path = os.path.join(args.checkpoint, "args.p")
    args = update_args(args, args_path)

    # Loading features
    with open(features_path, "rb") as pickle_file:
        feature_vocabs = pickle.load(pickle_file)

    data = add_feature_columns(
        data, check_repetition=args.use_repetitions, use_past=args.use_past,
    )

    data["features"] = data.apply(
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

    # Predictions
    tagger = pycrfsuite.Tagger()
    tagger.open(model_path)
    y_pred = crf_predict(tagger, data)

    data["speech_act"] = y_pred

    # Filter for important columns
    data_filtered = data.drop(
        columns=[
            "prev_tokens",
            "prev_speaker_code",
            "repeated_words",
            "nb_repwords",
            "ratio_repwords",
            "turn_length",
            "features",
        ]
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    os.makedirs(args.out, exist_ok=True)
    if not args.output_for_childes_db:
        out_path = os.path.join(args.out, "annotated.csv")
        data_filtered.to_csv(out_path, index=False)
    else:
        utterances_speech_acts_crf_path = os.path.join(args.out, "utterances_speech_acts_crf.csv")
        utterances_speech_acts_crf = data_filtered[["utterance_id", "speech_act"]]
        utterances_speech_acts_crf.to_csv(utterances_speech_acts_crf_path, index=False)

        metadata_path = os.path.join(args.out, "metadata.csv")
        metadata = pd.DataFrame.from_records([{"table_name": "utterances_speech_acts_crf_2024.1_1",
                                               "dataset_name": "speech_acts_crf",
                                               "entity_type": "utterances",
                                               "childes_db_version": "2021.1",
                                               "dataset_version": "1",
                                               "coding_table": "utterances_speech_acts_crf",
                                               "tag_type": "model",
                                               "model_version": "1",
                                               "date_of_release": "2024-04-01",
                                               "contact": "mitja.nikolaus@posteo.de",
                                               "citation": "https://doi.org/10.34842/2022.0532",
                                               }])
        metadata.to_csv(metadata_path, index=False)

        variables_path = os.path.join(args.out, "variables.csv")
        SPEECH_ACT_DESCRIPTIONS.to_csv(variables_path)

    if args.compare:
        data_children = data_filtered[data.speaker_code == CHILD]
        frequencies_children = calculate_frequencies(data_children["y_pred"].tolist())
        compare_frequencies(frequencies_children, args)
