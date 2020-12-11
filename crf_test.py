import os
import pickle
import argparse
from collections import Counter
import ast
from typing import Union, Tuple
from bidict import bidict

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
import pycrfsuite

from preprocess import SPEECH_ACT
from crf_train import (
    add_feature_columns,
    get_features_from_row,
    crf_predict,
    bio_classification_report,
)
from utils import SPEECH_ACT_UNINTELLIGIBLE, SPEECH_ACT_NO_FUNCTION


def load_training_args(args):
    # Load training arguments from metadata file
    text_file = open(os.path.join(args.model, "metadata.txt"), "r")
    lines = text_file.readlines()  # lines ending with "\n"
    for line in lines:
        arg_name, value = line[:-1].split(":\t")
        if arg_name in [
            "use_bi_grams",
            "use_action",
            "use_past",
            "use_repetitions",
            "use_past_actions",
            "use_pos",
            "match_age",
        ]:
            try:
                setattr(args, arg_name, ast.literal_eval(value))
            except ValueError as e:
                if "malformed node or string" in str(e):
                    setattr(args, arg_name, value)
            except Exception as e:
                raise e
    return args


def parse_args():
    argparser = argparse.ArgumentParser(description="Test a previously trained CRF")
    argparser.add_argument("data", type=str, help="file listing all dialogs")
    argparser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Ratio of given dataset to be used to testing",
    )
    argparser.add_argument(
        "--txt_columns",
        nargs="+",
        type=str,
        default=[],
        help=""".txt columns name (in order); most basic txt is ['spa_all', 'ut', 'time', 'speaker', 'sentence']""",
    )
    argparser.add_argument(
        "--match_age",
        type=int,
        nargs="+",
        default=None,
        help="ages to match data to - for split analysis",
    )
    argparser.add_argument(
        "--model",
        "-m",
        required=True,
        type=str,
        default=None,
        help="folder containing model, features and metadata",
    )
    # parameters for training/testing:
    argparser.add_argument(
        "--col_ages",
        type=str,
        default=None,
        help="if not None, plot evolution of accuracy over age groups",
    )
    argparser.add_argument(
        "--consistency_check",
        action="store_true",
        help="whether 'child' column matters in testing data.",
    )
    argparser.add_argument(
        "--prediction_mode",
        choices=["raw", "exclude_ool"],
        default="exclude_ool",
        type=str,
        help="Whether to predict with NOL/NAT/NEE labels or not.",
    )

    args = argparser.parse_args()

    return args


#### Report functions
def features_report(tagg):
    """Extracts weights and transitions learned from training
    From: https://github.com/scrapinghub/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb

    Input:
    --------
    tagg: pycrfsuite.Tagger

    Output:
    --------
    states: pd.DataFrame

    transitions: pd.DataFrame
    """
    info = tagg.info()
    # Feature weights
    state_features = Counter(info.state_features)
    states = pd.DataFrame(
        [
            {"weight": weight, "label": label, "attribute": attr}
            for (attr, label), weight in state_features.items()
        ]
    ).sort_values(by=["weight"], ascending=False)
    # Transitions
    trans_features = Counter(info.transitions)
    transitions = pd.DataFrame(
        [
            {"label_from": label_from, "label": label_to, "likelihood": weight}
            for (label_from, label_to), weight in trans_features.items()
        ]
    ).sort_values(by=["likelihood"], ascending=False)
    # return
    return states, transitions


def plot_testing(test_df: pd.DataFrame, file_location: str, col_ages):
    """Separating CHI/MOT and ages to plot accuracy, annotator agreement and number of categories over age."""
    tmp = []
    speakers = test_df["speaker"].unique().tolist()
    for age in sorted(test_df[col_ages].unique().tolist()):  # remove < 1Y?
        for spks in [[x] for x in speakers] + [speakers]:
            age_loc_sub = test_df[
                (test_df[col_ages] == age) & (test_df.speaker.isin(spks))
            ]
            acc = accuracy_score(age_loc_sub.y_true, age_loc_sub.y_pred, normalize=True)
            cks = cohen_kappa_score(age_loc_sub.y_true, age_loc_sub.y_pred)
            tmp.append(
                {
                    "age": age,
                    "locutor": "&".join(spks),
                    "accuracy": acc,
                    "agreement": cks,
                    "nb_labels": len(age_loc_sub.y_true.unique().tolist()),
                }
            )
        # also do CHI/MOT separately
    tmp = pd.DataFrame(tmp)
    speakers = tmp.locutor.unique().tolist()
    # plot
    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=(18, 10))
    for i, col in enumerate(["accuracy", "agreement", "nb_labels"]):
        for spks in speakers:
            ax[i].plot(
                tmp[tmp.locutor == spks].age, tmp[tmp.locutor == spks][col], label=spks
            )
            ax[i].set_ylabel(col)
    ax[2].set_xlabel("age (in months)")
    ax[2].legend()
    plt.savefig(file_location)


def report_to_file(dfs: dict, file_location: str):
    """Looping on each pd.DataFrame to log to excel"""
    writer = pd.ExcelWriter(file_location)
    for name, data in dfs.items():
        data.to_excel(writer, sheet_name=name)
    writer.save()


#### MAIN
if __name__ == "__main__":
    args = parse_args()
    args = load_training_args(args)
    print(args)

    # Loading model
    name = args.model
    if os.path.isdir(name):
        if name[-1] == "/":
            name = name[:-1]
    else:
        raise FileNotFoundError(f"Cannot find model {name}.")
    # update paths for input/output
    features_path = name + os.path.sep + "feature_vocabs.p"
    model_path = name + os.path.sep + "model.pycrfsuite"
    report_path = name + os.path.sep + args.data.replace("/", "_") + "_report.xlsx"
    plot_path = name + os.path.sep + args.data.split("/")[-1] + "_agesevol.png"
    classification_scores_path = name + os.path.sep + "classification_scores.p"

    # Loading data
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

    _, data_test = train_test_split(data, test_size=args.test_ratio, shuffle=False)
    print(f"Testing on {len(data_test)} utterances")

    # Loading features
    with open(features_path, "rb") as pickle_file:
        feature_vocabs = pickle.load(pickle_file)

    data_test["features"] = data_test.apply(
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

    grouped_test = data_test.groupby(by=["file_id"]).agg(
        {"features": lambda x: [y for y in x], "index": min}
    )

    y_pred = crf_predict(
        tagger,
        grouped_test.sort_values("index", ascending=True)["features"],
        mode=args.prediction_mode,
    )
    data_test["y_pred"] = [y for x in y_pred for y in x]  # flatten
    data_test["pred_OK"] = data_test.apply(
        lambda x: (x.y_pred == x[SPEECH_ACT]), axis=1
    )
    # Remove uninformative tags before doing analysis
    data_crf = data_test[~data_test[SPEECH_ACT].isin(["NAT", "NEE", SPEECH_ACT_UNINTELLIGIBLE, SPEECH_ACT_NO_FUNCTION])]
    # reports
    report, mat, acc, cks = bio_classification_report(
        data_crf[SPEECH_ACT].tolist(), data_crf["y_pred"].tolist()
    )
    states, transitions = features_report(tagger)

    int_cols = (
        ["file_id", "speaker"]
        + ([args.col_ages] if args.col_ages is not None else [])
        + [x for x in data_test.columns if "spa_" in x]
        + (["child"] if args.consistency_check else [])
        + [SPEECH_ACT, "y_pred", "pred_OK"]
    )

    report_d = {
        "test_data": data_crf[int_cols],
        "classification_report": report.T,
        "confusion_matrix": mat,
        "weights": states,
        "learned_transitions": transitions.pivot(
            index="label_from", columns="label", values="likelihood"
        ),
    }

    pickle.dump(report.T, open(classification_scores_path, "wb"))

    if args.col_ages is not None:
        plot_testing(data_test, plot_path, args.col_ages)

    # Write excel with all reports
    report_to_file(report_d, report_path)
