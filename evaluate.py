import argparse
import os
import pickle

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from torch.utils.data import DataLoader

from dataset import SpeechActsDataset
from generate_dataset import PADDING, SPEAKER_ADULT, SPEAKER_CHILD
from models import SpeechActLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_words(indices, vocab):
    return " ".join([vocab.itos[i] for i in indices if not vocab.itos[i] == PADDING])

def age_bin(age):
    """Return the corresponding age bin (14, 20 or 32) for a given age"""
    if age < 17:
        return 14
    elif age < 26:
        return 20
    else:
        return 32


def test(args):
    print("Start training with args: ", args)
    print("Device: ", device)
    # Load data
    vocab = pickle.load(open(args.data + "vocab.p", "rb"))
    label_vocab = pickle.load(open(args.data + "vocab_labels.p", "rb"))

    print("Loading data..")
    data = pd.read_hdf(args.data + os.sep + args.corpus, "test")

    dataset_test = SpeechActsDataset(data)

    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    print("Test samples: ", len(dataset_test))

    def evaluate(data_loader):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        all_true_labels = []
        all_predicted_labels = []
        all_ages = []
        speaker_is_child = []
        with torch.no_grad():
            for batch_id, (input_samples, targets, sequence_lengths, age) in enumerate(data_loader):
                # Move data to GPU
                targets = torch.tensor(targets).to(device)

                # Perform forward pass of the model
                predicted_labels = model.forward_decode(input_samples)
                predicted_labels = torch.tensor(predicted_labels).to(device)

                # Compare predicted labels to ground truth
                speaker_is_child += [True if x[0] == vocab.stoi[SPEAKER_CHILD] else False for x in input_samples]
                all_true_labels += targets.tolist()
                all_predicted_labels += predicted_labels.tolist()
                all_ages += [age] * len(input_samples)

                if args.verbose:
                    for i, (sample, label, predicted) in enumerate(zip(input_samples, targets, predicted_labels)):
                        if label_vocab.inverse[int(predicted)] != label_vocab.inverse[int(label)]:
                            print(
                                f"{get_words(sample, vocab)} Predicted: {label_vocab.inverse[int(predicted)]} True: {label_vocab.inverse[int(label)]}"
                            )


        acc = sum([p == l for p, l in zip(all_predicted_labels, all_true_labels)]) / len(all_true_labels)
        print("=" * 89)
        print("Test acc: {:5.2f}".format(acc))

        predicted_labels_adult = [label for label, is_child in zip(all_predicted_labels, speaker_is_child) if not is_child]
        true_labels_adult = [label for label, is_child in zip(all_true_labels, speaker_is_child) if not is_child]
        acc = np.equal(predicted_labels_adult, true_labels_adult).sum() / len(true_labels_adult)
        print("Test acc (adults' utterances): {:5.2f}".format(acc))

        predicted_labels_child = [label for label, is_child in zip(all_predicted_labels, speaker_is_child) if is_child]
        true_labels_child = [label for label, is_child in zip(all_true_labels, speaker_is_child) if is_child]
        acc = np.equal(predicted_labels_child, true_labels_child).sum() / len(true_labels_child)
        print("Test acc (children's utterances): {:5.2f}".format(acc))

        print("=" * 89)

        if args.verbose:
            all_ages = [age_bin(age) for age in all_ages]
            for age in [14, 20, 32]:
                print(f"Stats for age: {age} months")
                true_labels = [l for l, a in zip(all_true_labels, all_ages) if a == age]
                predicted_labels = [l for l, a in zip(all_predicted_labels, all_ages) if a == age]

                correct = np.equal(true_labels, predicted_labels).sum()
                total = len(true_labels)
                print("Acc: {:5.2f}".format(correct/total))
                print("#Samples: ",total)

                labels = label_vocab.keys()
                label_indices = [label_vocab[l] for l in labels]

                cm = confusion_matrix(true_labels, predicted_labels, normalize='true')
                print(cm)

                kappa = cohen_kappa_score(true_labels, predicted_labels)
                print("cohen's kappa: ", kappa)
                report = classification_report(true_labels, predicted_labels,  labels=label_indices, target_names=labels)
                print(report)


    # Load the saved model checkpoint.
    with open(args.checkpoint, "rb") as f:
        model = torch.load(f, map_location=device)

    # Run on test data.
    print("Eval:")
    evaluate(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
        help="location of the data corpus and vocabs",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="speech_acts_data_newengland.h5",
        help="name of the corpus file",
    )
    # TODO fix: works only with batch size one at the moment
    parser.add_argument(
        "--batch-size", type=int, default=1, metavar="N", help="batch size"
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model.pt",
        help="path to saved model checkpoint",
    )
    parser.add_argument('--verbose', '-v', action="store_true",
                           help="Increase verbosity")

    args = parser.parse_args()
    test(args)
