"""Generate datasets to be used by the ANNs (LSTM and Transformer)"""
import pickle
import argparse
from collections import Counter

import torch
import nltk
from bidict import bidict
from nltk import word_tokenize

import pandas as pd
import matplotlib.pyplot as plt

### Tag functions
from torchtext import vocab
from tqdm import tqdm

from utils import dataset_labels

import numpy as np

nltk.download("punkt")

PADDING = "<pad>"
UNKNOWN = "<unk>"
UNINTELLIGIBLE = "xxx"
PHONOLOGICAL = "yyy"
UNTRANSCRIBED = "www"
SPEAKER_CHILD = "<chi>"
SPEAKER_ADULT = "<adu>"

#### Read Data functions
def argparser():
    """Creating arparse.ArgumentParser and returning arguments"""
    argparser = argparse.ArgumentParser()
    # Data files
    argparser.add_argument("input_file", type=str, help="file listing dialogs")
    argparser.add_argument(
        "--vocab", "-v", required=False, help="path to existing vocab file"
    )
    argparser.add_argument(
        "--label-vocab", required=False, help="path to existing label vocab file"
    )
    argparser.add_argument(
        "--min-label-frequency", type=int, default=100, help="Minimum frequency to include label in vocab"
    )
    # Operations on data
    argparser.add_argument(
        "--out", type=str, default="data/", help="path for output files"
    )
    # parameters for training:
    argparser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="Maximum size of vocabulary",
    )

    args = argparser.parse_args()

    return args


def filter_labels(labels, min_frequency):
    """Filter labels to include only labels with a minimum number of occurrences"""
    counter = labels.value_counts()
    filtered = [l for l, c in counter.items() if c >= min_frequency]

    return filtered

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

def tokenize_sentence(sentence, speaker):
    # Tokenize sentence
    tokenized_sentence = word_tokenize(sentence)

    # Prepend speaker special token
    if speaker in ["MOT", "FAT", "INV"]:
        tokenized_sentence = [SPEAKER_ADULT] + tokenized_sentence
    elif speaker in ["CHI", "AMY"]:
        tokenized_sentence = [SPEAKER_CHILD] + tokenized_sentence
    else:
        raise RuntimeError("Unknown speaker code: ", speaker)

    return tokenized_sentence

if __name__ == "__main__":
    args = argparser()
    print(args)

    print("### Loading data:".upper())

    data = pd.read_csv(args.input_file, sep="\t", keep_default_na=False)
    data.reset_index(drop=False)
    data.rename(
        columns={col: col.lower() for col in data.columns}, inplace=True
    )
    target_label = "spa_2"

    tokenized_sentences = []
    ages = []
    data["tokens"] = data.apply(lambda x: tokenize_sentence(x.sentence, x.speaker), axis='columns')

    if args.vocab:
        vocab = pickle.load(open(args.vocab, "rb"))
    else:
        print("Building vocabulary..")
        vocab = build_vocabulary(data["tokens"])

    if args.label_vocab:
        label_vocab = pickle.load(open(args.label_vocab, "rb"))

    else:
        print("Building label vocabulary..")

        # Filter labels and replace all low frequent label names with UNK
        filtered_labels = filter_labels(data[target_label], args.min_label_frequency)
        filtered_labels.append("UNK")

        label_vocab = bidict({label: i for i, label in enumerate(filtered_labels)})
        pickle.dump(label_vocab, open(args.out + "vocab_labels.p", "wb"))

    data["labels"] = data[target_label].apply(lambda l: l if l in label_vocab.keys() else "UNK")

    # Convert words and labels to indices using the respective vocabs
    data["utterances"] = data.tokens.apply(lambda tokens: [vocab.stoi[t] for t in tokens])
    data["labels"] = data.labels.apply(lambda l: label_vocab[l])

    # Group by transcript (file name), each transcript is treated as one long input sequence
    grouped_train = data.groupby(by=['file_id']).agg({
        'utterances': lambda x: [y for y in x],
        'labels': lambda x: [y for y in x],
        'age_months': min,
    })

    dataset_type = ""
    if "train" in args.input_file:
        dataset_type = "train"
    elif "valid" in args.input_file:
        dataset_type = "val"
    elif "test" in args.input_file:
        dataset_type = "test"

    # data = pd.DataFrame(data={"utterance": utterances, "label": labels, "age": ages})
    print(grouped_train.head())
    grouped_train.to_hdf(args.out + "speech_acts_data.h5", key=dataset_type)
