import os
import pickle
import argparse
from collections import Counter
import ast
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    accuracy_score,
)
from scipy.stats import spearmanr

import pycrfsuite

from preprocess import SPEECH_ACT
from crf_train import (
    add_feature_columns,
    get_features_from_row,
    crf_predict,
    bio_classification_report,
)
from utils import (
    SPEECH_ACT_UNINTELLIGIBLE,
    SPEECH_ACT_NO_FUNCTION,
    make_train_test_splits,
    SPEECH_ACT_DESCRIPTIONS,
    calculate_frequencies,
    PATH_NEW_ENGLAND_UTTERANCES, ADULT,
)


def parse_args():
    argparser = argparse.ArgumentParser(description="Test a previously trained CRF")
    argparser.add_argument(
        "--data",
        default=PATH_NEW_ENGLAND_UTTERANCES,
        type=str,
        help="file listing all dialogs",
    )
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
        help="folder containing model and features",
    )
    # parameters for training/testing:
    argparser.add_argument(
        "--col_ages",
        type=str,
        default=None,
        help="if not None, plot evolution of accuracy over age groups",
    )
    argparser.add_argument(
        "--prediction_mode",
        choices=["raw", "exclude_ool"],
        default="raw",
        type=str,
        help="Whether to predict with NOL/NAT/NEE labels or not.",
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
    speakers = test_df["speaker_code"].unique().tolist()
    for age in sorted(test_df[col_ages].unique().tolist()):  # remove < 1Y?
        for spks in [[x] for x in speakers] + [speakers]:
            age_loc_sub = test_df[
                (test_df[col_ages] == age) & (test_df.speaker_code.isin(spks))
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
    print(args)

    # Loading model
    model_path = os.path.join(args.model, "model.pycrfsuite")
    features_path = os.path.join(args.model, "feature_vocabs.p")

    report_path = os.path.join(args.model, args.data.replace("/", "_") + "_report.xlsx")
    plot_path = os.path.join(args.model, args.data.split("/")[-1] + "_agesevol.png")
    classification_scores_path = os.path.join(args.model, "classification_scores.p")
    classification_scores_adult_path = os.path.join(args.model, "classification_scores_adult.p")

    # Loading data
    data = pd.read_pickle(args.data)

    data = add_feature_columns(
        data, check_repetition=args.use_repetitions, use_past=args.use_past,
    )

    data_train, data_test = make_train_test_splits(data, args.test_ratio)
    print(f"Testing on {len(data_test)} utterances")

    # Loading features
    with open(features_path, "rb") as pickle_file:
        feature_vocabs = pickle.load(pickle_file)

    data_test = data_test.assign(
        features=data_test.apply(
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

    y_pred = crf_predict(tagger, data_test, mode=args.prediction_mode,)
    data_test = data_test.assign(speech_act_predicted=y_pred)

    data_filtered = data_test.drop(
        columns=[
            "prev_tokens",
            "prev_speaker_code",
            "repeated_words",
            "nb_repwords",
            "ratio_repwords",
            "features",
        ]
    )
    data_filtered.to_pickle(os.path.join("checkpoints", "crf", "speech_acts.p"))

    data_test["pred_OK"] = data_test.apply(
        lambda x: (x.speech_act_predicted == x[SPEECH_ACT]), axis=1
    )
    # Remove uninformative tags before doing analysis
    data_crf = data_test[
        ~data_test[SPEECH_ACT].isin(
            ["NAT", "NEE", SPEECH_ACT_UNINTELLIGIBLE, SPEECH_ACT_NO_FUNCTION]
        )
    ]
    # reports
    report, confusion_matrix, acc, cks = bio_classification_report(
        data_crf[SPEECH_ACT].tolist(), data_crf["speech_act_predicted"].tolist()
    )
    states, transitions = features_report(tagger)

    int_cols = (
        ["transcript_file", "speaker_code"]
        + ([args.col_ages] if args.col_ages is not None else [])
        + [x for x in data_test.columns if "spa_" in x]
        + [SPEECH_ACT, "speech_act_predicted", "pred_OK"]
    )

    report_d = {
        "test_data": data_crf[int_cols],
        "classification_report": report.T,
        "confusion_matrix": confusion_matrix,
        "weights": states,
        "learned_transitions": transitions.pivot(
            index="label_from", columns="label", values="likelihood"
        ),
    }

    pickle.dump(report.T, open(classification_scores_path, "wb"))

    data_crf_adult = data_crf[data_crf.speaker_code == ADULT]

    cr_adult = classification_report(
        data_crf_adult[SPEECH_ACT].tolist(),
        data_crf_adult["speech_act_predicted"].tolist(),
        digits=3,
        output_dict=True,
        zero_division=0,
    )
    pickle.dump(pd.DataFrame(cr_adult).T, open(classification_scores_adult_path, "wb"))

    confusion_matrix = confusion_matrix.T
    for label in np.unique(data_test[SPEECH_ACT]):
        confusions = confusion_matrix[confusion_matrix[label] > 0.05].index.values
        confusions = np.delete(confusions, np.where(confusions == label))

        # label_category = SPEECH_ACT_DESCRIPTIONS.loc[label]["Category"]
        # genuine_confusions = []
        # for confusion in confusions:
        #     confused_label_category = SPEECH_ACT_DESCRIPTIONS.loc[confusion]["Category"]
        #     if label_category == confused_label_category:
        #         genuine_confusions.append(confusion)

        if len(confusions) > 0 and label in SPEECH_ACT_DESCRIPTIONS.Description:
            print(
                f"{label} ({SPEECH_ACT_DESCRIPTIONS.Description[label]}) is confused with:"
            )
            for confusion in confusions:
                print(confusion, SPEECH_ACT_DESCRIPTIONS.Description[confusion])
            print("")

    if args.col_ages is not None:
        plot_testing(data_test, plot_path, args.col_ages)

    report = classification_report(
        data_crf[SPEECH_ACT].tolist(),
        data_crf["speech_act_predicted"].tolist(),
        digits=3,
        zero_division=0,
    )
    print(report)

    report_dict = classification_report(
        data_crf[SPEECH_ACT].tolist(),
        data_crf["speech_act_predicted"].tolist(),
        labels=sorted(data_crf[SPEECH_ACT].unique()),
        digits=3,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T
    pd.set_option("display.float_format", lambda x: "%.3f" % x)
    print(report_df.to_latex())

    # Get training data label frequencies:
    freqs = dict(calculate_frequencies(data_train[SPEECH_ACT]))
    filtered_freqs = [
        freqs[speech_act] for speech_act in report_df.index if speech_act in freqs
    ]
    filtered_f1 = [
        f1 for speech_act, f1 in report_df["f1-score"].items() if speech_act in freqs
    ]
    corr, p_value = spearmanr(filtered_f1, filtered_freqs)

    print(f"Spearman correlation between freq and f-score: {corr:.2f} (p = {p_value})")

    # Write excel with all reports
    report_to_file(report_d, report_path)
