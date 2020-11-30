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

def preprend_speaker_token(tokens, speaker):
    """Prepend speaker special token"""
    if speaker in ["MOT", "FAT", "INV"]:
        tokens = [SPEAKER_ADULT] + tokens
    elif speaker in ["CHI", "AMY"]:
        tokens = [SPEAKER_CHILD] + tokens
    else:
        raise RuntimeError("Unknown speaker code: ", speaker)

    return tokens




def tokenize_sentence(sentence, speaker):
    # Tokenize sentence
    tokenized_sentence = word_tokenize(sentence)

    tokenized_sentence = preprend_speaker_token(tokenized_sentence)

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

    label_vocab = dataset_labels(target_label.upper())
    pickle.dump(label_vocab, open(args.out + "vocab_labels.p", "wb"))

    # Convert words and labels to indices using the respective vocabs
    data["utterances"] = data.tokens.apply(lambda tokens: [vocab.stoi[t] for t in tokens])
    data["labels"] = data[target_label].apply(lambda l: label_vocab[l])

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
