import os
import pickle
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from scipy.stats import entropy
import pycrfsuite

import seaborn as sns

from sklearn.model_selection import KFold

from preprocess import SPEECH_ACT
from utils import SPEECH_ACT_DESCRIPTIONS
from crf_train import (
    add_feature_columns,
    get_features_from_row,
    generate_features_vocabs, crf_predict, plot_training,
)
from crf_test import bio_classification_report, report_to_file

AGE_MONTHS_GROUPS = {
    14: [13, 14, 15],
    20: [18, 19, 20, 21],
    32: [27, 28, 29, 30, 31, 32, 33],
}

def argparser():
    argparser = argparse.ArgumentParser(
        description="Perform cross-validation for the CRF",
    )
    # Data files
    argparser.add_argument("--data", type=str, help="file listing all dialogs")
    # Operations on data
    argparser.add_argument(
        "--age", type=int, default=None, help="filter data for children's age"
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
    argparser.add_argument('--use-pos', '-pos', action='store_true', help="whether to add POS tags to features")
    argparser.add_argument(
        "--use-action",
        "-act",
        action="store_true",
        help="whether to use action features to train the algorithm, if they are in the data",
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

    # Definitions
    number_segments_length_feature = 10

    print("### Loading data:".upper())

    data = pd.read_pickle(args.data)

    data = add_feature_columns(
        data,
        use_action=args.use_action,
        check_repetition=args.use_repetitions,
        use_past=args.use_past,
        use_pastact=args.use_past_actions,
        use_pos=args.use_pos,
    )

    # Location for weight save
    checkpoint_path = "checkpoints/crf_cross_validation/"
    print("Saving model at: {}".format(checkpoint_path))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    logger = {}  # Dictionary containing results
    freport = {}  # Dictionary containing reports
    counters = {}


    # Filter data by childen's age
    if args.age:
        data = data[data.age_months.isin(AGE_MONTHS_GROUPS[args.age])]

    # Gather ground-truth label distributions:
    data_children = data[data.speaker == "CHI"]
    counts = Counter(data_children[SPEECH_ACT])
    observed_labels = [k for k in SPEECH_ACT_DESCRIPTIONS.Name.keys() if counts[k] > 0]
    counters["gold"] = dict.fromkeys(observed_labels)
    counters["gold"].update((k, counts[k]) for k in counts.keys() & observed_labels)
    for k in counters["gold"].keys():
        counters["gold"][k] /= len(data_children)

    pickle.dump(
        counters["gold"], open(f"data/frequencies_gold_age_{str(args.age)}.p", "wb")
    )

    counts_predicted = Counter()

    # Split data
    kf = KFold(n_splits=args.num_splits, random_state=0)

    file_names = data["file_id"].unique().tolist()
    for i, (train_indices, test_indices) in enumerate(kf.split(file_names)):
        train_files = [file_names[i] for i in train_indices]
        test_files = [file_names[i] for i in test_indices]

        data_train = data[data["file_id"].isin(train_files)]
        data_test = data[data["file_id"].isin(test_files)]

        print(
            f"\n### Training on permutation {i} - {len(data_train)} utterances in train,  {len(data_test)} utterances in test set: "
        )
        nm = os.path.join(checkpoint_path, f"permutation_{i}")

        # generating features
        features_idx = generate_features_vocabs(
            data_train,
            args.nb_occurrences,
            args.use_bi_grams,
            args.use_action,
            args.use_repetitions,
            args.use_pos,
            bin_cut=number_segments_length_feature,
        )

        # creating crf features set for train
        data_train["features"] = data_train.apply(
            lambda x: get_features_from_row(
                features_idx,
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

        # Once the features are done, groupby name and extract a list of lists
        grouped_train = (
            data_train.dropna(subset=[SPEECH_ACT])
            .groupby(by=["file_id"])
            .agg(
                {
                    "features": lambda x: [y for y in x],
                    SPEECH_ACT: lambda x: [y for y in x],
                    "index": min,
                }
            )
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

        data_test["features"] = data_test.apply(
            lambda x: get_features_from_row(
                features_idx,
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

        data_test.dropna(subset=[SPEECH_ACT], inplace=True)
        X_dev = data_test.groupby(by=["file_id"]).agg(
            {"features": lambda x: [y for y in x], "index": min}
        )
        y_pred = crf_predict(
            tagger,
            X_dev.sort_values("index", ascending=True)["features"],
            mode=args.prediction_mode,
        )
        data_test["y_pred"] = [y for x in y_pred for y in x]  # flatten
        data_test["y_true"] = data_test[SPEECH_ACT]
        data_test["pred_OK"] = data_test.apply(lambda x: (x.y_pred == x.y_true), axis=1)
        # remove ['NOL', 'NAT', 'NEE'] for prediction and reports
        data_crf = data_test[~data_test["y_true"].isin(["NOL", "NAT", "NEE"])]
        # reports
        report, mat, acc, cks = bio_classification_report(
            data_crf["y_true"].tolist(), data_crf["y_pred"].tolist()
        )
        logger[i] = acc
        freport[i] = {"report": report, "cm": mat}

        # Filter for children's utterances
        data_crf_children = data_crf[data_crf.speaker == "CHI"]

        counts = Counter(data_crf_children["y_pred"].tolist())
        counts_predicted.update(counts)

    counters["pred"] = dict.fromkeys(observed_labels)
    counters["pred"].update(
        (k, counts_predicted[k]) for k in counts_predicted.keys() & observed_labels
    )

    for k in counters["pred"].keys():
        if counters["pred"][k]:
            counters["pred"][k] /= len(data_children)
        else:
            counters["pred"][k] = 0

    labels = observed_labels * 2
    splits = np.concatenate([[str(i)] * len(observed_labels) for i in counters.keys()])
    counts = np.concatenate([list(counter.values()) for counter in counters.values()])
    df = pd.DataFrame(
        zip(labels, splits, counts), columns=["speech_act", "split", "frequency"]
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(x="speech_act", hue="split", y="frequency", data=df)
    plt.show()

    kl_divergence = entropy(
        list(counters["pred"].values()), qk=list(counters["gold"].values())
    )
    print(f"KL Divergence: {kl_divergence:.3f}")

    train_per = pd.Series(logger, name="acc_over_train_percentage")

    report_to_file(
        {
            **{"report_" + str(n): d["report"].T for n, d in freport.items()},
            **{"cm_" + str(n): d["cm"].T for n, d in freport.items()},
        },
        os.path.join(checkpoint_path, "report.xlsx"),
    )

    with open(os.path.join(checkpoint_path, "metadata.txt"), "w") as meta_file:  # dumping metadata
        for arg in vars(args):
            meta_file.write("{0}:\t{1}\n".format(arg, getattr(args, arg)))
        meta_file.write("{0}:\t{1}\n".format("Experiment", "Datasets"))

    print("Average accuracy over all splits: ", np.average(list(logger.values())))
