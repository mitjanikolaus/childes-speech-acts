"""Generate datasets to be used by the ANNs (LSTM and Transformer)"""
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
    argparser = argparse.ArgumentParser()
    # Data files
    argparser.add_argument("input_file", type=str, help="file listing dialogs")
    argparser.add_argument(
        "--vocab", "-v", required=False, help="path to existing vocab file"
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

    print("### Loading data:".upper())

    data_train = pd.read_csv(args.input_file, sep="\t", keep_default_na=False)
    data_train.reset_index(drop=False)
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

    data = pd.DataFrame(data={"features": features, "labels": labels})
    print(data.head())
    data.to_hdf(args.out + "speech_acts_data.h5", key=dataset_type)
