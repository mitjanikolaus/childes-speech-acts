import os
import argparse
import pickle
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble, svm
import pycrfsuite

from crf_train import (
    add_feature_columns,
    generate_features_vocabs,
    get_features_from_row,
    bio_classification_report, get_n_grams,
)
from preprocess import SPEECH_ACT, ADULT
from utils import (
    SPEECH_ACT_UNINTELLIGIBLE,
    SPEECH_ACT_NO_FUNCTION,
    TRAIN_TEST_SPLIT_RANDOM_STATE,
    make_train_test_splits,
    dataset_labels,
)


def argparser():
    argparser = argparse.ArgumentParser(
        description="Train a Baseline model and test it.",
    )
    # Data files
    argparser.add_argument("data", type=str, help="file listing train dialogs")
    # Operations on data
    argparser.add_argument(
        "--model",
        type=str,
        choices=["SVC", "LSVC", "RF"],
        default="RF",
        help="which algorithm to use for baseline: SVM (classifier ou linear classifier), RandomForest(100 trees)",
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
        "--test-ratio",
        type=float,
        default=0.2,
        help="Ratio of dataset to be used to testing",
    )
    argparser.add_argument(
        "--use-bi-grams",
        "-bi",
        action="store_true",
        help="whether to use bi-gram features to train the algorithm",
    )
    argparser.add_argument(
        "--use-action",
        "-act",
        action="store_true",
        help="whether to use action features to train the algorithm, if they are in the data",
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
        "--use-past-actions",
        "-pa",
        action="store_true",
        help="whether to add actions from the previous sentence to features",
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
    features: dict, tokens: list, speaker: str, prev_speaker: str, ln: int, use_bi_grams, **kwargs
):
    """Replacing input list tokens with feature index. """
    nb_features = (
        max([max([int(x) for x in v.values()]) for v in features.values()]) + 1
    )
    # list features
    features_sparse = [
        features["words"][w] for w in tokens if w in features["words"].keys()
    ]  # words
    features_sparse.append(1 if speaker == ADULT else 0)
    features_sparse.append(1 if prev_speaker == ADULT else 0)

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

    if ("action_tokens" in kwargs) and (
        kwargs["action_tokens"] is not None
    ):  # actions are descriptions just like 'words'
        features_sparse += [
            features["action"][w]
            for w in kwargs["action_tokens"]
            if w in features["action"].keys()
        ]
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
    ):  # actions are descriptions just like 'words'
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
    args = argparser()
    print(args)

    # Definitions
    number_segments_length_feature = 10

    print("### Loading data:".upper())

    data = pd.read_pickle(args.data)

    data = add_feature_columns(
        data,
        use_action=args.use_action,
        match_age=args.match_age,
        check_repetition=args.use_repetitions,
        use_past=args.use_past,
        use_pastact=args.use_past_actions,
        use_pos=args.use_pos,
    )

    data_train, data_test = make_train_test_splits(data, args.test_ratio)

    print("### Creating features:")
    feature_vocabs = generate_features_vocabs(
        data_train,
        args.nb_occurrences,
        args.use_bi_grams,
        args.use_action,
        args.use_repetitions,
        args.use_pos,
        bin_cut=number_segments_length_feature,
    )

    # creating features set for train
    X = data_train.apply(
        lambda x: get_baseline_features_from_row(
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
    model = baseline_model(args.model, weights, True)  # Taking imbalance into account
    model.fit(X, y)
    # dump(mdl, os.path.join(name, 'baseline.joblib'))

    X_test = data_test.apply(
        lambda x: get_baseline_features_from_row(
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

    # transforming
    X_test = np.array(X_test.tolist())

    y_pred = [labels.inverse[x] for x in model.predict(X_test)]

    # y_pred = baseline_predict(bs_model, X, labels, mode=args.prediction_mode)
    data_test["y_pred"] = y_pred
    data_test["pred_OK"] = data_test.apply(lambda x: (x.y_pred == x[SPEECH_ACT]), axis=1)
    data_bs = data_test[~data_test[SPEECH_ACT].isin(["NOL", "NAT", "NEE"])]

    report, _, _, _ = bio_classification_report(
        data_bs[SPEECH_ACT].tolist(), data_bs["y_pred"].tolist()
    )

    print(report.T)
    path = os.path.join("results","baseline")
    os.makedirs(path, exist_ok=True)
    pickle.dump(report.T, open(os.path.join(path, f"classification_scores_{args.model}.p"), "wb"))

