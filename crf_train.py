import os
import argparse
import pickle
from collections import Counter
import json
from itertools import tee, islice
from typing import Union, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from nltk import ngrams
from sklearn.metrics import (
    accuracy_score,
)
from sklearn.model_selection import train_test_split
import pycrfsuite

from preprocess import SPEECH_ACT, ADULT

def argparser():
    argparser = argparse.ArgumentParser(
        description="Train a CRF and test it.",
    )
    # Data files
    argparser.add_argument("data", type=str, help="file listing train dialogs")
    # Operations on data
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
        default=.2,
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
        "--use-past_actions",
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


#### Features functions
def add_feature_columns(
    p: pd.DataFrame,
    match_age: Union[str, list] = None,
    use_action: bool = False,
    use_past: bool = False,
    use_pastact: bool = False,
    check_repetition: bool = False,
):
    """Function adding features to the data:
    * turn_length
    * tags (if necessary): extract interchange/illocutionary from general tag
    * action_tokens (if necessary): splitting action sentence into individual words
    * age_months: matching age to experimental labels
    * repeted_words:
    * number of repeated words
    * ratio of words that were repeated from previous sentence over sentence length
    """
    # sentence: using tokens to count & all
    p["tokens"] = p.tokens
    p["turn_length"] = p.tokens.apply(len)

    # action: creating action tokens
    if use_action:
        p["action"].fillna("", inplace=True)
        p["action_tokens"] = p.action.apply(lambda x: x.lower().split())

    # matching age with theoretical age from the study
    if "age_months" in p.columns and match_age is not None:
        match_age = match_age if isinstance(match_age, list) else [match_age]
        p["age_months"] = p.age_months.apply(
            lambda age: min(match_age, key=lambda x: abs(x - age))
        )

    # repetition features
    if check_repetition or use_past or use_pastact:
        p["prev_file"] = p.file_id.shift(1).fillna(p.file_id.iloc[0])
        p["prev_spk"] = p.speaker.shift(1).fillna(p.speaker.iloc[0])
        p["prev_st"] = p.tokens.shift(
            1
        )
        p["prev_st"].iloc[0] = p.tokens.iloc[0]

    if check_repetition:
        p["repeated_words"] = p.apply(
            lambda x: [w for w in x.tokens if w in x.prev_st]
            if (x.prev_spk != x.speaker) and (x.file_id == x.prev_file)
            else [],
            axis=1,
        )
        p["nb_repwords"] = p.repeated_words.apply(len)
        p["ratio_repwords"] = p.nb_repwords / p.turn_length

    if use_past:
        p["past"] = p.apply(
            lambda x: x.prev_st if (x.file_id == x.prev_file) else [], axis=1
        )

    if use_action and use_pastact:
        p["prev_act"] = p["action_tokens"].shift(1)
        p["prev_act"].iloc[0] = p["action_tokens"].iloc[0]
        p["past_act"] = p.apply(
            lambda x: x.prev_act if (x.file_id == x.prev_file) else [], axis=1
        )
    # remove helper columns
    p = p.drop(columns=["prev_spk", "prev_st", "prev_file", "prev_act"], errors="ignore")

    # return Dataframe
    return p


def get_features_from_row(
    features: dict, tokens: list, speaker: str, ln: int, use_bi_grams, **kwargs
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
    action_tokens: `list`
            data action if actions are not taken into account

    past_tokens: `list`

    pastact_tokens: `list`

    repetitions: `Tuple[list, float, float]`
            contains the list of repeated words, number of words repeated, ratio of repeated words over sequence

    Output:
    -------
    feat_glob: `dict`
            dictionary of same shape as feature, but only containing features relevant to data line
    """
    feat_glob = {}

    # TODO: add 'UNK' token
    feat_glob["words"] = Counter([w for w in tokens if (w in features["words"].keys())])

    feat_glob["speaker"] = 1.0 if speaker == ADULT else 0.0
    feat_glob["length"] = {
        k: (1 if ln < float(k.split("-")[1]) and ln >= float(k.split("-")[0]) else 0)
        for k in features["length_bins"].keys()
    }

    if use_bi_grams:
        bi_grams = ["-".join(n_gram) for n_gram in get_n_grams(tokens, 2) if n_gram in features["bigrams"].keys()]
        feat_glob["bigrams"] = Counter(bi_grams)

    if ("action_tokens" in kwargs) and (kwargs["action_tokens"] is not None):
        # actions are descriptions just like 'words'
        feat_glob["actions"] = Counter(
            [w for w in kwargs["action_tokens"] if (w in features["action"].keys())]
        )  # if (features['action'] is not None) else Counter(action_tokens)
    if ("repetitions" in kwargs) and (kwargs["repetitions"] is not None):
        (rep_words, len_rep, ratio_rep) = kwargs["repetitions"]
        feat_glob["repeated_words"] = Counter(
            [w for w in rep_words if (w in features["words"].keys())]
        )
        feat_glob["rep_length"] = {
            k: (
                1
                if len_rep <= float(k.split("-")[1])
                and len_rep >= float(k.split("-")[0])
                else 0
            )
            for k in features["rep_length_bins"].keys()
        }
        feat_glob["rep_ratio"] = {
            k: (
                1
                if ratio_rep <= float(k.split("-")[1])
                and ratio_rep >= float(k.split("-")[0])
                else 0
            )
            for k in features["rep_ratio_bins"].keys()
        }
    if ("past_tokens" in kwargs) and (kwargs["past_tokens"] is not None):
        feat_glob["past"] = Counter(
            [w for w in kwargs["past_tokens"] if (w in features["words"].keys())]
        )
    if ("pastact_tokens" in kwargs) and (kwargs["pastact_tokens"] is not None):
        feat_glob["past_actions"] = Counter(
            [w for w in kwargs["pastact_tokens"] if (w in features["action"].keys())]
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
    use_action: bool,
    use_repetitions: bool,
    bin_cut: int = 10,
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
    feature_vocabs["words"] = {k: i for i, k in enumerate(sorted(count_vocabulary.keys()))}
    print("\nThere are {} words in the vocab".format(len(feature_vocabs["words"])))

    # Features: sentence length (+ logging counts)
    data["len_bin"], bins = pd.qcut(
        data.turn_length, q=bin_cut, duplicates="drop", labels=False, retbins=True
    )

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
        feature_vocabs["bigrams"] = {k: nb_feat+i for i, k in enumerate(sorted(bi_grams_vocab.keys()))}

        print("\nMost common bigrams: ", bi_grams_counter.most_common(20))
        print("There are {} bigrams in the vocab".format(len(feature_vocabs["bigrams"])))

    # Features: actions
    if use_action:
        count_actions = [y for x in data.action_tokens.tolist() for y in x]  # flatten
        count_actions = dict(Counter(count_actions))
        # filtering features
        count_actions = {k: v for k, v in count_actions.items() if v > nb_occ}
        # turning vocabulary into numbered features - ordered vocabulary
        nb_feat = max([max(v.values()) for v in feature_vocabs.values()])
        feature_vocabs["action"] = {
            k: i + nb_feat for i, k in enumerate(sorted(count_actions.keys()))
        }
        print("\nThere are {} words in the actions".format(len(feature_vocabs["action"])))

    # Features: repetitions of previous utterance
    if use_repetitions:
        nb_feat = max([max(v.values()) for v in feature_vocabs.values()])
        # features esp for length & ratio - repeated words can use previously defined features
        # lengths
        _, bins = pd.qcut(
            data.nb_repwords, q=bin_cut, duplicates="drop", labels=False, retbins=True
        )
        feature_vocabs["rep_length_bins"] = {
            "{}-{}".format(k, bins[i + 1]): (nb_feat + i)
            for i, k in enumerate(bins[:-1])
        }
        # ratios
        _, bins = pd.qcut(
            data.ratio_repwords,
            q=bin_cut,
            duplicates="drop",
            labels=False,
            retbins=True,
        )
        feature_vocabs["rep_ratio_bins"] = {
            "{}-{}".format(k, bins[i + 1]): (nb_feat + i)
            for i, k in enumerate(bins[:-1])
        }
        print("\nRepetition ratio splits: ")
        for i, k in enumerate(bins[:-1]):
            print("\tlabel {}: turns of length {}-{}".format(i, k, bins[i + 1]))

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
    gp_data: list,
    mode: str = "raw",
    exclude_labels: list = ["NOL", "NAT", "NEE"],
) -> Union[list, Tuple[list, pd.DataFrame]]:
    """Return predictions for the test data, grouped by file. 3 modes for return:
            * Return raw predictions (raw)
            * Return predictions with only valid tags (exclude_ool)
            * Return predictions (valid tags) and probabilities for each class (rt_proba)

    Predictions are returned unflattened

    https://python-crfsuite.readthedocs.io/en/latest/pycrfsuite.html
    """
    if mode not in ["raw", "exclude_ool", "rt_proba"]:
        raise ValueError(
            f"mode must be one of raw|exclude_ool|rt_proba; currently {mode}"
        )
    if mode == "raw":
        return [tagger.tag(xseq) for xseq in gp_data]
    labels = tagger.labels()

    res = []
    y_pred = []
    for fi, xseq in enumerate(gp_data):
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
        file_proba["file_id"] = fi
        res.append(file_proba)

    if mode == "rt_proba":
        return y_pred, pd.concat(res, axis=0)
    return y_pred  # else


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
    )

    data_train, data_test = train_test_split(data, test_size=args.test_ratio, shuffle=False)

    print("### Creating features:")
    feature_vocabs = generate_features_vocabs(
        data_train,
        args.nb_occurrences,
        args.use_bi_grams,
        args.use_action,
        args.use_repetitions,
        bin_cut=number_segments_length_feature,
    )

    # creating crf features set for train
    data_train["features"] = data_train.apply(
        lambda x: get_features_from_row(
            feature_vocabs,
            x.tokens,
            x["speaker"],
            x.turn_length,
            use_bi_grams = args.use_bi_grams,
            action_tokens=None if not args.use_action else x.action_tokens,
            repetitions=None
            if not args.use_repetitions
            else (x.repeated_words, x.nb_repwords, x.ratio_repwords),
            past_tokens=None if not args.use_past else x.past,
            pastact_tokens=None if not args.use_past_actions else x.past_act,
        ),
        axis=1,
    )

    # Once the features are done, groupby name and extract a list of lists
    # The list contains transcripts, which each contain a list of utterances
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
    )

    grouped_train = sklearn.utils.shuffle(grouped_train)

    print("\n### Training starts.".upper())
    trainer = pycrfsuite.Trainer(verbose=args.verbose)
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
    if args.verbose:
        plot_training(trainer, checkpoint_path)

    # dumping features
    with open(os.path.join(checkpoint_path, "feature_vocabs.p"), "wb") as pickle_file:
        pickle.dump(feature_vocabs, pickle_file)

    # dumping metadata
    with open(os.path.join(checkpoint_path, "metadata.txt"), "w") as meta_file:
        for arg in vars(args):
            meta_file.write("{0}:\t{1}\n".format(arg, getattr(args, arg)))

    # Calculate test accuracy
    tagger = pycrfsuite.Tagger()
    tagger.open(os.path.join(checkpoint_path, "model.pycrfsuite"))

    data_test["features"] = data_test.apply(
        lambda x: get_features_from_row(
            feature_vocabs,
            x.tokens,
            x["speaker"],
            x.turn_length,
            use_bi_grams=args.use_bi_grams,
            action_tokens=None if not args.use_action else x.action_tokens,
            repetitions=None
            if not args.use_repetitions
            else (x.repeated_words, x.nb_repwords, x.ratio_repwords),
            past_tokens=None if not args.use_past else x.past,
            pastact_tokens=None if not args.use_past_actions else x.past_act,
        ),
        axis=1,
    )

    # Once the features are done, groupby name and extract a list of lists
    # The list contains transcripts, which each contain a list of utterances
    grouped_test = (
        data_test.dropna(subset=[SPEECH_ACT])
            .groupby(by=["file_id"])
            .agg(
            {
                "features": lambda x: [y for y in x],
                SPEECH_ACT: lambda x: [y for y in x],
                "index": min,
            }
        )
    )

    y_pred = crf_predict(
        tagger,
        grouped_test.sort_values("index", ascending=True)["features"],
        mode="exclude_ool",
    )
    data_test["y_pred"] = [y for x in y_pred for y in x]  # flatten
    data_crf = data_test[~data_test[SPEECH_ACT].isin(["NOL", "NAT", "NEE"])]

    print(data_crf[SPEECH_ACT].tolist()[:30])
    print(data_crf["y_pred"].tolist()[:30])

    acc = accuracy_score(
        data_crf[SPEECH_ACT].tolist(), data_crf["y_pred"].tolist(), normalize=True
    )
    print(f"Accuracy on test set: {acc:.3f}")
