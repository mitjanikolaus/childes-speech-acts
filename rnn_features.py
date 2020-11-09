import pickle
import argparse
from collections import Counter

import torch
import nltk
from nltk import word_tokenize

import pandas as pd
import matplotlib.pyplot as plt

### Tag functions
from torchtext import vocab
from tqdm import tqdm

from utils import dataset_labels

nltk.download("punkt")

PADDING = "<pad>"
UNKNOWN = "<unk>"
SPEAKER_CHILD = "<chi>"
SPEAKER_ADULT = "<adu>"


#### Read Data functions
def argparser():
    """Creating arparse.ArgumentParser and returning arguments"""
    argparser = argparse.ArgumentParser(
        description="Train a CRF and test it.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # Data files
    argparser.add_argument("input_file", type=str, help="file listing dialogs")
    argparser.add_argument(
        "--vocab", "-v", required=False, help="path to existing vocab file"
    )
    argparser.add_argument(
        "--txt_columns",
        nargs="+",
        type=str,
        default=[],
        help=""".txt columns name (in order); most basic txt is ['spa_all', 'ut', 'time', 'speaker', 'sentence']""",
    )
    # Operations on data
    argparser.add_argument(
        "--match_age",
        type=int,
        nargs="+",
        default=None,
        help="ages to match data to - for split analysis",
    )
    argparser.add_argument(
        "--keep_tag",
        choices=["all", "1", "2", "2a"],
        default="all",
        help="keep first part / second part / all tag",
    )
    argparser.add_argument(
        "--cut",
        type=int,
        default=1000000,
        help="if specified, use the first n train dialogs instead of all.",
    )
    argparser.add_argument(
        "--out", type=str, default="data/", help="path for output files"
    )
    # parameters for training:
    argparser.add_argument(
        "--nb_occurrences",
        "-noc",
        type=int,
        default=5,
        help="number of minimum occurrences for word to appear in features",
    )
    argparser.add_argument(
        "--use_past",
        "-past",
        action="store_true",
        help="whether to add previous sentence as features",
    )
    argparser.add_argument(
        "--use_repetitions",
        "-rep",
        action="store_true",
        help="whether to check in data if words were repeated from previous sentence, to train the algorithm",
    )
    argparser.add_argument(
        "--use_past_actions",
        "-pa",
        action="store_true",
        help="whether to add actions from the previous sentence to features",
    )
    argparser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to display training iterations output.",
    )
    # Baseline model
    argparser.add_argument(
        "--baseline",
        type=str,
        choices=["SVC", "LSVC", "NB", "RF"],
        default=None,
        help="which algorithm to use for baseline: SVM (classifier ou linear classifier), NaiveBayes, RandomForest(100 trees)",
    )
    argparser.add_argument(
        "--balance_ex",
        action="store_true",
        help="whether to take proportion of each class into account when training (imbalanced dataset).",
    )

    args = argparser.parse_args()

    return args


def build_vocabulary(data):
    word_counter = Counter()
    for tokens in data:
        word_counter.update(tokens)
    print(f"Vocab: {word_counter.most_common(100)}")
    print(f"Total number of words length: {len(word_counter)}")
    vocabulary = vocab.Vocab(
        word_counter,
        max_size=10000,
        specials=[PADDING, SPEAKER_CHILD, SPEAKER_ADULT, UNKNOWN],
    )
    pickle.dump(vocabulary, open(args.out + "vocab.p", "wb"))

    return vocabulary


#### MAIN
if __name__ == "__main__":
    args = argparser()
    print(args)

    # Definitions
    number_words_for_feature = args.nb_occurrences  # default 5
    number_segments_length_feature = 10
    # number_segments_turn_position = 10 # not used for now
    target_label = "spa_" + args.keep_tag

    print("### Loading data:".upper())

    data_train = pd.read_csv(
        args.input_file, sep="\t", keep_default_na=False
    ).reset_index(drop=False)
    data_train.rename(
        columns={col: col.lower() for col in data_train.columns}, inplace=True
    )
    target_label = [x for x in data_train.columns if "spa_" in x][0]
    args.training_tag = target_label

    # limit number of samples for development purposes
    # data_train = data_train.head()

    tokenized_sentences = []
    for i, row in tqdm(data_train.iterrows(), total=data_train.shape[0]):
        # Tokenize sentence
        tokenized_sentence = word_tokenize(row["sentence"])
        # Prepend speaker special token
        if row["speaker"] in ["MOT", "FAT", "INV"]:
            tokenized_sentence = [SPEAKER_ADULT] + tokenized_sentence
        elif row["speaker"] in ["CHI"]:
            tokenized_sentence = [SPEAKER_CHILD] + tokenized_sentence
        else:
            raise RuntimeError("Unknown speaker code: ", row["speaker"])

        tokenized_sentences.append(tokenized_sentence)

    if args.vocab:
        vocab = pickle.load(open(args.vocab, "rb"))
    else:
        print("Building vocabulary..")
        vocab = build_vocabulary(tokenized_sentences)

    label_vocab = dataset_labels(target_label.upper())
    pickle.dump(label_vocab, open(args.out + "vocab_labels.p", "wb"))

    features = []
    labels = []
    for tokens, label in zip(tokenized_sentences, data_train[target_label].to_list()):
        features.append([vocab.stoi[t] for t in tokens])
        labels.append(label_vocab[label])

    dataset_type = ""
    if "train" in args.input_file:
        dataset_type = "train"
    elif "valid" in args.input_file:
        dataset_type = "val"
    elif "test" in args.input_file:
        dataset_type = "test"

    data = pd.DataFrame(zip(features, labels))
    print(data.head())
    data.to_hdf(args.out + "speech_acts_data.h5", key=dataset_type)
