import os
import pickle
import argparse

import numpy as np
import pandas as pd
import sklearn
import pycrfsuite

from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold

from utils import (
    TRAIN_TEST_SPLIT_RANDOM_STATE,
    SPEECH_ACT,
    PATH_NEW_ENGLAND_UTTERANCES_ANNOTATED,
)
from crf_train import (
    add_feature_columns,
    get_features_from_row,
    generate_features_vocabs,
    crf_predict,
)


def argparser():
    argparser = argparse.ArgumentParser(
        description="Perform cross-validation for the CRF",
    )
    # Data files
    argparser.add_argument("--data", type=str, help="file listing all dialogs")
    # parameters for training:
    argparser.add_argument(
        "--nb-occurrences",
        "-noc",
        type=int,
        default=5,
        help="number of minimum occurrences for word to appear in features",
    )
    argparser.add_argument(
        "--num-splits",
        type=int,
        default=5,
        help="number of splits to perform crossvalidation over",
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
    argparser.add_argument(
        "--prediction-mode",
        choices=["raw", "exclude_ool"],
        default="exclude_ool",
        type=str,
        help="Whether to predict with NOL/NAT/NEE labels or not.",
    )

    args = argparser.parse_args()
    return args


if __name__ == "__main__":
    args = argparser()
    print(args)

    print("### Loading data:".upper())

    data = pd.read_pickle(args.data)

    data = add_feature_columns(
        data, check_repetition=args.use_repetitions, use_past=args.use_past,
    )

    # Location for weight save
    checkpoint_path = "checkpoints/crf_cross_validation/"
    print("Saving model at: {}".format(checkpoint_path))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Split data
    kf = KFold(
        n_splits=args.num_splits,
        shuffle=True,
        random_state=TRAIN_TEST_SPLIT_RANDOM_STATE,
    )

    accuracies = []
    result_dataframes = []

    file_names = data["transcript_file"].unique().tolist()
    for i, (train_indices, test_indices) in enumerate(kf.split(file_names)):
        train_files = [file_names[i] for i in train_indices]
        test_files = [file_names[i] for i in test_indices]

        data_train = data[data["transcript_file"].isin(train_files)]
        data_test = data[data["transcript_file"].isin(test_files)]

        print(
            f"\n### Training on permutation {i} - {len(data_train)} utterances in train,  {len(data_test)} utterances in test set: "
        )
        nm = os.path.join(checkpoint_path, f"permutation_{i}")

        # generating features
        features_idx = generate_features_vocabs(
            data_train,
            args.nb_occurrences,
            args.use_bi_grams,
            args.use_repetitions,
            args.use_pos,
        )

        # creating crf features set for train
        data_train = data_train.assign(
            features=data_train.apply(
                lambda x: get_features_from_row(
                    features_idx,
                    x.tokens,
                    x["speaker_code"],
                    x["prev_speaker_code"],
                    x.turn_length,
                    use_bi_grams=args.use_bi_grams,
                    repetitions=None
                    if not args.use_repetitions
                    else (x.repeated_words, x.nb_repwords, x.ratio_repwords),
                    prev_tokens=None if not args.use_past else x.past,
                    pos_tags=None if not args.use_pos else x.pos,
                ),
                axis=1,
            )
        )

        # Once the features are done, groupby name and extract a list of lists
        grouped_train = data_train.groupby(by=["transcript_file"]).agg(
            {
                "features": lambda x: [y for y in x],
                SPEECH_ACT: lambda x: [y for y in x],
            }
        )  # listed by apparition order
        grouped_train = sklearn.utils.shuffle(grouped_train)

        ### Training
        trainer = pycrfsuite.Trainer(verbose=args.verbose)
        # Adding data
        for idx, file_data in grouped_train.iterrows():
            trainer.append(
                file_data["features"], file_data[SPEECH_ACT]
            )  # X_train, y_train
        # Parameters
        trainer.set_params(
            {
                "c1": 1,  # coefficient for L1 penalty
                "c2": 1e-3,  # coefficient for L2 penalty
                "max_iterations": 50,  # stop earlier
                "feature.possible_transitions": True,  # include transitions that are possible, but not observed
            }
        )
        print("Saving model at: {}".format(nm))

        trainer.train(nm + "_model.pycrfsuite")
        with open(nm + "_features.p", "wb") as pickle_file:  # dumping features
            pickle.dump(features_idx, pickle_file)

        ### Testing
        tagger = pycrfsuite.Tagger()
        tagger.open(nm + "_model.pycrfsuite")

        data_test = data_test.assign(
            features=data_test.apply(
                lambda x: get_features_from_row(
                    features_idx,
                    x.tokens,
                    x["speaker_code"],
                    x["prev_speaker_code"],
                    x.turn_length,
                    use_bi_grams=args.use_bi_grams,
                    repetitions=None
                    if not args.use_repetitions
                    else (x.repeated_words, x.nb_repwords, x.ratio_repwords),
                    prev_tokens=None if not args.use_past else x.past,
                    pos_tags=None if not args.use_pos else x.pos,
                ),
                axis=1,
            )
        )

        data_test.dropna(subset=[SPEECH_ACT], inplace=True)
        X_dev = data_test.groupby(by=["transcript_file"]).agg(
            {"features": lambda x: [y for y in x]}
        )
        y_pred = crf_predict(tagger, X_dev["features"], mode=args.prediction_mode,)
        data_test["y_pred"] = [y for x in y_pred for y in x]  # flatten
        data_test["pred_OK"] = data_test.apply(
            lambda x: (x.y_pred == x[SPEECH_ACT]), axis=1
        )

        # Remove uninformative tags before doing analysis
        data_crf = data_test[~data_test[SPEECH_ACT].isin(["NAT", "NEE"])]

        acc = accuracy_score(data_crf[SPEECH_ACT].tolist(), data_crf["y_pred"].tolist())

        accuracies.append(acc)
        result_dataframes.append(
            data_crf[
                [
                    "utterance_id",
                    "transcript_file",
                    "speaker_code",
                    "age",
                    "tokens",
                    SPEECH_ACT,
                    "y_pred",
                ]
            ]
        )

    print("mean accuracy over all splits: ", np.average(accuracies))
    print("std accuracy over all splits: ", np.std(accuracies))

    result_dataframe = result_dataframes[0]
    for df in result_dataframes[1:]:
        result_dataframe = result_dataframe.append(df)
    pickle.dump(result_dataframe, open(PATH_NEW_ENGLAND_UTTERANCES_ANNOTATED, "wb"))
