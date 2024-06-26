import os
import argparse
import pickle
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble, svm
import pycrfsuite
from sklearn.model_selection import KFold

from crf_train import (
    add_feature_columns,
    generate_features_vocabs,
    get_features_from_row,
    bio_classification_report,
    get_n_grams,
)
from preprocess import SPEECH_ACT, ADULT
from utils import (
    SPEECH_ACT_UNINTELLIGIBLE,
    SPEECH_ACT_NO_FUNCTION,
    TRAIN_TEST_SPLIT_RANDOM_STATE,
    make_train_test_splits,
    dataset_labels, PATH_NEW_ENGLAND_UTTERANCES,
)


def parse_args():
    argparser = argparse.ArgumentParser(
        description="Train a Baseline model and test it.",
    )
    # Data files
    argparser.add_argument("--data", type=str, default=PATH_NEW_ENGLAND_UTTERANCES, help="file listing train dialogs")
    # Operations on data
    argparser.add_argument(
        "--model",
        type=str,
        choices=["SVC", "LSVC", "RF"],
        default="RF",
        help="which algorithm to use for baseline: SVM (classifier ou linear classifier), RandomForest(100 trees)",
    )
    argparser.add_argument(
        "--num-splits",
        type=int,
        default=5,
        help="number of splits to perform crossvalidation over",
    )
    argparser.add_argument(
        "--match-age",
        type=int,
        nargs="+",
        default=None,
        help="ages to match data to - for split analysis",
    )
    # parameters for training:
    argparser.add_argument(
        "--nb-occurrences",
        "-noc",
        type=int,
        default=5,
        help="number of minimum occurrences for word to appear in features",
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
    argparser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to display training iterations output.",
    )

    args = argparser.parse_args()

    return args


def baseline_model(name: str, weights: dict, balance: bool):
    """Create and update (if need be) model with weights"""
    models = {
        "SVC": svm.SVC(),
        "LSVC": svm.LinearSVC(),
        "RF": ensemble.RandomForestClassifier(n_estimators=100),
    }
    if balance:
        try:
            models[name].set_params(class_weight=weights)
        except ValueError as e:
            if "Invalid parameter class_weight for estimator" in str(
                e
            ):  # GaussianNB has no such parameter for instance
                pass
            else:
                raise e
        except Exception as e:
            raise e
    return models[name]


def get_baseline_features_from_row(
    features: dict,
    tokens: list,
    speaker_code: str,
    prev_speaker_code: str,
    ln: int,
    use_bi_grams,
    **kwargs,
):
    """Replacing input list tokens with feature index. """
    nb_features = (
        max([max([int(x) for x in v.values()]) for v in features.values()]) + 1
    )
    # list features
    features_sparse = [
        features["words"][w] for w in tokens if w in features["words"].keys()
    ]  # words
    features_sparse.append(1 if speaker_code == ADULT else 0)
    features_sparse.append(1 if prev_speaker_code == ADULT else 0)

    for k in features["length_bins"].keys():  # length
        if ln <= float(k.split("-")[1]) and ln >= float(k.split("-")[0]):
            features_sparse.append(features["length_bins"][k])

    if use_bi_grams:
        bi_grams = [
            "-".join(n_gram)
            for n_gram in get_n_grams(tokens, 2)
            if n_gram in features["bigrams"].keys()
        ]
        features_sparse = [
            features["bi_grams"][b] for b in bi_grams if b in features["bigrams"].keys()
        ]  # words

    if ("repetitions" in kwargs) and (
        kwargs["repetitions"] is not None
    ):  # not using words, only ratio+len
        (_, len_rep, ratio_rep) = kwargs["repetitions"]
        for k in features["rep_length_bins"].keys():
            if len_rep <= float(k.split("-")[1]) and len_rep >= float(k.split("-")[0]):
                features_sparse.append(features["rep_length_bins"][k])
        for k in features["rep_ratio_bins"].keys():
            if len_rep <= float(k.split("-")[1]) and len_rep >= float(k.split("-")[0]):
                features_sparse.append(features["rep_ratio_bins"][k])

    if ("pos_tags" in kwargs) and (
        kwargs["pos_tags"] is not None
    ):
        features_sparse += [
            features["pos"][w]
            for w in kwargs["pos_tags"]
            if w in features["pos"].keys()
        ]

    # transforming features
    features_full = [1 if i in features_sparse else 0 for i in range(nb_features)]

    return features_full


#### MAIN
if __name__ == "__main__":
    args = parse_args()
    print(args)

    print("### Loading data:".upper())

    data = pd.read_pickle(args.data)

    data = add_feature_columns(
        data,
        check_repetition=args.use_repetitions,
        use_past=args.use_past,
    )

    accuracies = []

    # Split data
    kf = KFold(n_splits=args.num_splits, shuffle=True, random_state=TRAIN_TEST_SPLIT_RANDOM_STATE)

    file_names = data["transcript_file"].unique().tolist()
    for i, (train_indices, test_indices) in enumerate(kf.split(file_names)):
        train_files = [file_names[i] for i in train_indices]
        test_files = [file_names[i] for i in test_indices]

        data_train = data[data["transcript_file"].isin(train_files)]
        data_test = data[data["transcript_file"].isin(test_files)]

        print(
            f"\n### Training on permutation {i} - {len(data_train)} utterances in train,  {len(data_test)} utterances in test set: "
        )

        print("### Creating features:")
        feature_vocabs = generate_features_vocabs(
            data_train,
            args.nb_occurrences,
            args.use_bi_grams,
            args.use_repetitions,
            args.use_pos,
        )

        # creating features set for train
        X = data_train.apply(
            lambda x: get_baseline_features_from_row(
                feature_vocabs,
                x.tokens,
                x["speaker_code"],
                x["prev_speaker_code"],
                x.turn_length,
                use_bi_grams=args.use_bi_grams,
                repetitions=None
                if not args.use_repetitions
                else (x.repeated_words, x.ratio_repwords),
                past_tokens=None if not args.use_past else x.past,
                pos_tags=None if not args.use_pos else x.pos,
            ),
            axis=1,
        )

        y = data_train[SPEECH_ACT].tolist()
        weights = dict(Counter(y))
        # ID from label - bidict
        labels = dataset_labels(add_empty_labels=True)
        # transforming
        X = np.array(X.tolist())
        y = np.array([labels[lab] for lab in y])  # to ID
        weights = {
            labels[lab]: v / len(y) for lab, v in weights.items()
        }  # update weights as proportion, ID as labels
        model = baseline_model(
            args.model, weights, True
        )  # Taking imbalance into account
        model.fit(X, y)
        # dump(mdl, os.path.join(name, 'baseline.joblib'))

        X_test = data_test.apply(
            lambda x: get_baseline_features_from_row(
                feature_vocabs,
                x.tokens,
                x["speaker_code"],
                x["prev_speaker_code"],
                x.turn_length,
                use_bi_grams=args.use_bi_grams,
                repetitions=None
                if not args.use_repetitions
                else (x.repeated_words, x.ratio_repwords),
                past_tokens=None if not args.use_past else x.past,
                pos_tags=None if not args.use_pos else x.pos,
            ),
            axis=1,
        )

        # transforming
        X_test = np.array(X_test.tolist())

        y_pred = [labels.inverse[x] for x in model.predict(X_test)]

        # y_pred = baseline_predict(bs_model, X, labels, mode=args.prediction_mode)
        data_test["y_pred"] = y_pred
        data_test["pred_OK"] = data_test.apply(
            lambda x: (x.y_pred == x[SPEECH_ACT]), axis=1
        )
        data_bs = data_test[~data_test[SPEECH_ACT].isin(["NOL", "NAT", "NEE"])]

        report, _, acc, _ = bio_classification_report(
            data_bs[SPEECH_ACT].tolist(), data_bs["y_pred"].tolist()
        )

        accuracies.append(acc)

        path = os.path.join("results", "baseline")
        os.makedirs(path, exist_ok=True)
        pickle.dump(
            report.T,
            open(os.path.join(path, f"classification_scores_{args.model}.p"), "wb"),
        )

    print("Mean acc: ", np.mean(accuracies))
    print("std acc: ", np.std(accuracies))
