import os
import argparse
import pickle
from collections import Counter
from typing import Union, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from nltk import ngrams
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
)
import numpy as np
import pycrfsuite

from preprocess import SPEECH_ACT
from utils import (
    SPEECH_ACT_UNINTELLIGIBLE,
    SPEECH_ACT_NO_FUNCTION,
    make_train_test_splits,
    PUNCTUATION,
    UNKNOWN,
    PATH_NEW_ENGLAND_UTTERANCES,
    CHILD,
)


def parse_args():
    argparser = argparse.ArgumentParser(description="Train a CRF and test it.",)
    # Data files
    argparser.add_argument(
        "--data",
        type=str,
        default=PATH_NEW_ENGLAND_UTTERANCES,
        help="file listing train dialogs",
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
        "--cut-train-set",
        type=float,
        default=1.0,
        help="Reduce number of examples in the training set to the given fraction",
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


#### Features functions
def add_feature_columns(
    data: pd.DataFrame, use_past: bool = False, check_repetition: bool = False,
):
    """Function adding features to the data:
    * turn_length
    * tags (if necessary): extract interchange/illocutionary from general tag
    * repeated_words:
    * number of repeated words
    * ratio of words that were repeated from previous sentence over sentence length
    """
    data["tokens"] = data.tokens

    data["turn_length"] = data.tokens.apply(len)

    data["prev_file"] = data.transcript_file.shift(
        1, fill_value=data.transcript_file.iloc[0]
    )

    data["prev_speaker_code"] = data["speaker_code"].shift(1)
    data.loc[data.transcript_file != data.prev_file, "prev_speaker_code"] = None

    # repetition features
    if check_repetition or use_past:
        data["prev_tokens"] = data.tokens.shift(1)
        data.loc[data.transcript_file != data.prev_file, "prev_tokens"] = None
        data["prev_tokens"] = data["prev_tokens"].fillna("").apply(list)

    if check_repetition:
        data["repeated_words"] = data.apply(
            lambda x: [
                w
                for w in x.tokens
                if w in x.prev_tokens and w not in PUNCTUATION.values()
            ]
            if (x.prev_speaker_code != x.speaker_code)
            else [],
            axis=1,
        )
        data["nb_repwords"] = data.repeated_words.apply(len)
        data["ratio_repwords"] = data.nb_repwords / data.turn_length

    # remove helper columns
    data = data.drop(columns=["prev_file"])

    # return Dataframe
    return data


def get_features_from_row(
    features: dict,
    tokens: list,
    speaker: str,
    prev_speaker: str,
    ln: int,
    use_bi_grams,
    **kwargs,
):
    """Replacing input list tokens with feature index


    Input:
    -------
    features: `dict`
            dictionary of all features used, by type: {'words':Counter(), ...}

    spoken_tokens: `list`
            data sentence

    speaker: `str`
            MOT/CHI

    ln: `int`
            sentence length

    Kwargs:
    --------
    prev_tokens: `list`

    repetitions: `Tuple[list, float, float]`
            contains the list of repeated words, number of words repeated, ratio of repeated words over sequence

    Output:
    -------
    feat_glob: `dict`
            dictionary of same shape as feature, but only containing features relevant to data line
    """
    feat_glob = {}

    feat_glob["words"] = Counter(
        [w if w in features["words"].keys() else UNKNOWN for w in tokens]
    )

    feat_glob["speaker_code"] = 1 if speaker == CHILD else 0
    feat_glob["speaker_changed"] = 1 if speaker != prev_speaker else 0

    feat_glob["length"] = {
        k: 1
        for k in features["length_bins"].keys()
        if float(k.split("-")[1]) > ln >= float(k.split("-")[0])
    }

    if use_bi_grams:
        bi_grams = [
            "-".join(n_gram)
            for n_gram in get_n_grams(tokens, 2)
            if n_gram in features["bigrams"].keys()
        ]
        feat_glob["bigrams"] = Counter(bi_grams)

    if ("repetitions" in kwargs) and (kwargs["repetitions"] is not None):
        (rep_words, ratio_rep) = kwargs["repetitions"]
        feat_glob["repeated_words"] = Counter(
            [w for w in rep_words if (w in features["words"].keys())]
        )
        feat_glob["rep_ratio"] = {
            k: 1
            for k in features["rep_ratio_bins"].keys()
            if float(k.split("-")[1]) >= ratio_rep >= float(k.split("-")[0])
        }
    if ("prev_tokens" in kwargs) and (kwargs["prev_tokens"] is not None):
        feat_glob["prev_tokens"] = Counter(
            [w for w in kwargs["prev_tokens"] if (w in features["words"].keys())]
        )

    if ("pos_tags" in kwargs) and (kwargs["pos_tags"] is not None):
        feat_glob["pos"] = Counter(
            [w for w in kwargs["pos_tags"] if (w in features["pos"].keys())]
        )

    return feat_glob


def get_n_grams(utterance, n):
    # Cut off punctuation
    utterance = utterance[:-1]
    n_grams = ngrams(utterance, n)
    return n_grams


def get_n_grams_counter(utterances, n):
    counter = Counter()
    for utterance in utterances:
        n_grams = get_n_grams(utterance, n)
        counter.update(n_grams)
    return counter


def generate_features_vocabs(
    data: pd.DataFrame,
    nb_occ: int,
    use_bi_grams: bool,
    use_repetitions: bool,
    use_pos: bool,
    num_bins_length: int = 5,
    num_bins_rep=3,
) -> dict:
    """Analyse data according to arguments passed and generate features_idx dictionary. Printing log data to console."""
    feature_vocabs = {}
    print("\nTag counts: ")
    count_tags = data[SPEECH_ACT].value_counts().to_dict()
    for k in sorted(count_tags.keys()):
        print("{}: {}".format(k, count_tags[k]), end=" ; ")

    # Features: vocabulary (spoken)
    count_vocabulary = [y for x in data.tokens.tolist() for y in x]
    count_vocabulary = dict(Counter(count_vocabulary))
    count_vocabulary = {k: v for k, v in count_vocabulary.items() if v > nb_occ}

    # turning vocabulary into numbered features - ordered vocabulary
    all_words = sorted(count_vocabulary.keys()) + [UNKNOWN]
    feature_vocabs["words"] = {k: i for i, k in enumerate(all_words)}
    print("\nThere are {} words in the vocab".format(len(feature_vocabs["words"])))

    # Features: sentence length (+ logging counts)
    length_bin, bins = pd.qcut(
        data.turn_length,
        q=num_bins_length,
        duplicates="drop",
        labels=False,
        retbins=True,
    )
    data = data.assign(len_bin=length_bin)

    print("\nTurn length splits: ")
    for i, k in enumerate(bins[:-1]):
        print("\tlabel {}: turns of length {}-{}".format(i, k, bins[i + 1]))

    nb_feat = max([max(v.values()) for v in feature_vocabs.values()])
    feature_vocabs["length_bins"] = {
        "{}-{}".format(k, bins[i + 1]): (nb_feat + i) for i, k in enumerate(bins[:-1])
    }
    feature_vocabs["length"] = {i: (nb_feat + i) for i, _ in enumerate(bins[:-1])}

    if use_bi_grams:
        bi_grams_counter = get_n_grams_counter(data.tokens.tolist(), 2)
        bi_grams_vocab = {k: v for k, v in dict(bi_grams_counter).items() if v > nb_occ}
        nb_feat = max([max(v.values()) for v in feature_vocabs.values()])
        feature_vocabs["bigrams"] = {
            k: nb_feat + i for i, k in enumerate(sorted(bi_grams_vocab.keys()))
        }

        print("\nMost common bigrams: ", bi_grams_counter.most_common(20))
        print(
            "There are {} bigrams in the vocab".format(len(feature_vocabs["bigrams"]))
        )

    # Features: repetitions of previous utterance
    if use_repetitions:
        nb_feat = max([max(v.values()) for v in feature_vocabs.values()])
        # features esp for length & ratio - repeated words can use previously defined features
        # lengths
        # ratios
        _, bins = pd.cut(
            data.ratio_repwords,
            bins=num_bins_rep,
            duplicates="drop",
            labels=False,
            retbins=True,
        )
        bins = [max(0, k) for k in bins]
        feature_vocabs["rep_ratio_bins"] = {
            "{}-{}".format(k, bins[i + 1]): (nb_feat + i)
            for i, k in enumerate(bins[:-1])
        }
        print("\nRepetition ratio splits: ")
        for i, k in enumerate(bins[:-1]):
            print("\tlabel {}: repetitions of {}-{}%".format(i, k, bins[i + 1]))

    if use_pos:
        nb_feat = max([max(v.values()) for v in feature_vocabs.values()])

        pos_vocab = Counter()
        for tags in data.pos.tolist():
            if tags:
                pos_vocab.update(tags)

        pos_vocab = dict(pos_vocab)
        # filtering features
        pos_vocab = {k: v for k, v in pos_vocab.items() if v > nb_occ}
        # turning vocabulary into numbered features - ordered vocabulary
        feature_vocabs["pos"] = {
            k: i + nb_feat for i, k in enumerate(sorted(pos_vocab.keys()))
        }
        print(
            "\nThere are {} pos tags in the features:".format(
                len(feature_vocabs["pos"])
            )
        )
        print(list(feature_vocabs["pos"].keys()))

    return feature_vocabs


### REPORT
def plot_training(trainer, file_name):
    logs = pd.DataFrame(trainer.logparser.iterations)  # initially list of dicts
    # columns: {'loss', 'error_norm', 'linesearch_trials', 'active_features', 'num', 'time', 'scores', 'linesearch_step', 'feature_norm'}
    # FYI scores is empty

    logs.set_index("num", inplace=True)
    for col in ["loss", "active_features"]:
        plt.figure()
        plt.plot(logs[col])
        plt.savefig(file_name + "/" + col + ".png")


def crf_predict(
    tagger: pycrfsuite.Tagger,
    data: pd.DataFrame,
    mode: str = "raw",
    exclude_labels: list = ["NOL", "NAT", "NEE"],
) -> Union[list, Tuple[list, pd.DataFrame]]:
    """Return predictions for the test data, grouped by file. 3 modes for return:
            * Return raw predictions (raw)
            * Return predictions with only valid tags (exclude_ool)

    Predictions are returned unflattened

    https://python-crfsuite.readthedocs.io/en/latest/pycrfsuite.html
    """
    grouped_data = data.groupby(by=["transcript_file"], sort=False).agg(
        {"features": lambda x: [y for y in x]}
    )["features"]

    if mode not in ["raw", "exclude_ool"]:
        raise ValueError(
            f"mode must be one of raw|exclude_ool|rt_proba; currently {mode}"
        )
    if mode == "raw":
        y_pred = [tagger.tag(xseq) for xseq in grouped_data]
    else:
        labels = tagger.labels()

        y_pred = []
        for fi, xseq in enumerate(grouped_data):
            tagger.set(xseq)
            file_proba = pd.DataFrame(
                {
                    label: [tagger.marginal(label, i) for i in range(len(xseq))]
                    for label in labels
                }
            )
            y_pred.append(
                file_proba[[col for col in file_proba.columns if col not in exclude_labels]]
                .idxmax(axis=1)
                .tolist()
            )
            file_proba["transcript_file"] = fi

    return [y for x in y_pred for y in x]  # flatten


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    Requires scikit-learn 0.20+

    Output:
    --------
    cr: pd.DataFrame

    cm: np.array

    acc: float
    """
    cr = classification_report(
        y_true, y_pred, digits=3, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    acc = accuracy_score(y_true, y_pred, normalize=True)
    cks = cohen_kappa_score(y_true, y_pred)

    print("==> Accuracy: {0:.3f}".format(acc))
    print(
        "==> Cohen Kappa Score: {0:.3f} \t(pure chance: {1:.3f})".format(
            cks, 1.0 / len(set(y_true))
        )
    )
    # using both as index in case not the same labels in it
    return (
        pd.DataFrame(cr),
        pd.DataFrame(
            cm, index=sorted(set(y_true + y_pred)), columns=sorted(set(y_true + y_pred))
        ),
        acc,
        cks,
    )


def train(
    data_file,
    use_bi_grams=False,
    use_repetitions=False,
    use_past=False,
    use_pos=False,
    test_ratio=0.2,
    cut_train_set=1.0,
    nb_occurrences=5,
    verbose=False,
):
    print("### Loading data:".upper())

    data = pd.read_pickle(data_file)

    data = add_feature_columns(
        data, check_repetition=use_repetitions, use_past=use_past
    )

    data_train, data_test = make_train_test_splits(data, test_ratio)
    print("Number of samples in train split: ", len(data_train))
    print("Number of samples in test split: ", len(data_test))

    if cut_train_set < 1.0:
        train_files = data_train["transcript_file"].unique().tolist()
        train_subset = np.random.choice(
            len(train_files), size=int(len(train_files) * cut_train_set), replace=False
        )
        train_files = [train_files[x] for x in train_subset]
        data_train = data_train[data_train["transcript_file"].isin(train_files)]

    print("### Creating features:")
    feature_vocabs = generate_features_vocabs(
        data_train, nb_occurrences, use_bi_grams, use_repetitions, use_pos,
    )

    # creating crf features set for train
    data_train = data_train.assign(
        features=data_train.apply(
            lambda x: get_features_from_row(
                feature_vocabs,
                x.tokens,
                x.speaker_code,
                x.prev_speaker_code,
                x.turn_length,
                use_bi_grams=use_bi_grams,
                repetitions=None
                if not use_repetitions
                else (x.repeated_words, x.ratio_repwords),
                prev_tokens=None if not use_past else x.prev_tokens,
                pos_tags=None if not use_pos else x.pos,
            ),
            axis=1,
        )
    )

    # Once the features are done, groupby name and extract a list of lists
    # The list contains transcripts, which each contain a list of utterances
    grouped_train = data_train.groupby(by=["transcript_file"]).agg(
        {"features": lambda x: [y for y in x], SPEECH_ACT: lambda x: [y for y in x],}
    )

    grouped_train = sklearn.utils.shuffle(grouped_train)

    print("\n### Training starts.".upper())
    trainer = pycrfsuite.Trainer(verbose=verbose)
    # Add data
    for idx, file_data in grouped_train.iterrows():
        trainer.append(file_data["features"], file_data[SPEECH_ACT])  # X_train, y_train

    # Parameters
    trainer.set_params(
        {
            "c1": 1,  # coefficient for L1 penalty
            "c2": 1e-3,  # coefficient for L2 penalty
            "max_iterations": 50,  # stop earlier
            "feature.possible_transitions": True,  # include transitions that are possible, but not observed
        }
    )

    # Location for weight save
    checkpoint_path = "checkpoints/crf/"
    print("Saving model at: {}".format(checkpoint_path))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    trainer.train(os.path.join(checkpoint_path, "model.pycrfsuite"))

    # plotting training curves
    if verbose:
        plot_training(trainer, checkpoint_path)

    # dumping features
    with open(os.path.join(checkpoint_path, "feature_vocabs.p"), "wb") as pickle_file:
        pickle.dump(feature_vocabs, pickle_file)

    # Calculate test accuracy
    tagger = pycrfsuite.Tagger()
    tagger.open(os.path.join(checkpoint_path, "model.pycrfsuite"))

    data_test = data_test.assign(
        features=data_test.apply(
            lambda x: get_features_from_row(
                feature_vocabs,
                x.tokens,
                x["speaker_code"],
                x["prev_speaker_code"],
                x.turn_length,
                use_bi_grams=use_bi_grams,
                repetitions=None
                if not use_repetitions
                else (x.repeated_words, x.ratio_repwords),
                prev_tokens=None if not use_past else x.prev_tokens,
                pos_tags=None if not use_pos else x.pos,
            ),
            axis=1,
        )
    )

    data_test["y_pred"] = crf_predict(tagger, data_test)

    # Remove uninformative tags before doing analysis
    data_crf = data_test[
        ~data_test[SPEECH_ACT].isin(
            ["NAT", "NEE", SPEECH_ACT_UNINTELLIGIBLE, SPEECH_ACT_NO_FUNCTION]
        )
    ]

    _, _, acc, _ = bio_classification_report(
        data_crf[SPEECH_ACT].tolist(), data_crf["y_pred"].tolist()
    )

    return acc, len(data_train)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(
        args.data,
        args.use_bi_grams,
        args.use_repetitions,
        args.use_past,
        args.use_pos,
        args.test_ratio,
        args.cut_train_set,
        args.nb_occurrences,
        args.verbose,
    )
