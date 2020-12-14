import argparse
import os
import pickle

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, plot_confusion_matrix
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from nn_dataset import SpeechActsDataset
from nn_train import prepare_data
from utils import TRAIN_TEST_SPLIT_RANDOM_STATE, make_train_test_splits, get_words
from utils import SPEECH_ACT_DESCRIPTIONS, SPEAKER_CHILD, preprend_speaker_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PUNCTUATION = [".","!","?"]

def test(args):
    print("Start evaluation with args: ", args)
    print("Device: ", device)

    print("Loading data..")
    data = pd.read_pickle(args.data)

    vocab = pickle.load(open(os.path.join(args.model,"vocab.p"), "rb"))
    label_vocab = pickle.load(open(os.path.join(args.model, "vocab_labels.p"), "rb"))

    _, data_test = make_train_test_splits(data, args.test_ratio)

    data_test = prepare_data(data_test, vocab, label_vocab)

    dataset_test = SpeechActsDataset(data_test)

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


        acc = int(sum([p == l for p, l in zip(all_predicted_labels, all_true_labels)])) / len(all_true_labels)
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
            # all_ages = [age_bin(age) for age in all_ages]
            # for age in [14, 20, 32]:
            #     print(f"Stats for age: {age} months")
            #     true_labels = [l for l, a in zip(all_true_labels, all_ages) if a == age]
            #     predicted_labels = [l for l, a in zip(all_predicted_labels, all_ages) if a == age]
            predicted_labels = all_predicted_labels
            true_labels = all_true_labels

            correct = np.equal(true_labels, predicted_labels).sum()
            total = len(true_labels)
            print("Acc: {:5.2f}".format(correct/total))
            print("#Samples: ",total)

            labels = label_vocab.keys()
            label_indices = [label_vocab[l] for l in labels]

            cm = confusion_matrix(true_labels, predicted_labels, normalize='true', labels=list(label_vocab.values()))

            for label_idx in label_vocab.inverse.keys():
                confusions = [label_vocab.inverse[i] for i in np.where(cm[label_idx] > .1)[0] if i != label_idx]
                label = label_vocab.inverse[label_idx]
                if confusions:
                    print(f"{label} ({SPEECH_ACT_DESCRIPTIONS.Description[label]}) is confused with:")
                    for confusion in confusions:
                        print(confusion, SPEECH_ACT_DESCRIPTIONS.Description[confusion])
                    print("")



            fig, ax = plt.subplots(1, 1)
            ax.imshow(cm, cmap='binary', interpolation='None')
            ax.set_xticks(np.arange(len(label_vocab)))
            ax.set_xticklabels(label_vocab.keys())
            ax.set_yticks(np.arange(len(label_vocab)))
            ax.set_yticklabels(label_vocab.keys())
            plt.xticks(rotation=90)
            plt.grid()
            plt.show()


            kappa = cohen_kappa_score(true_labels, predicted_labels)
            print("cohen's kappa: ", kappa)
            report = classification_report(true_labels, predicted_labels,  labels=label_indices, target_names=labels)
            print(report)


    # Load the saved model checkpoint.
    with open(os.path.join(args.model, "model.pt"), "rb") as f:
        model = torch.load(f, map_location=device)

    # Run on test data.
    print("Eval:")
    evaluate(test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/new_england_preprocessed.p",
        help="path to the data corpus",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data/",
        help="directory of the model checkpoint and vocabs",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Ratio of dataset to be used to testing",
    )
    # TODO fix: works only with batch size one at the moment
    parser.add_argument(
        "--batch-size", type=int, default=1, metavar="N", help="batch size"
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument('--verbose', '-v', action="store_true",
                           help="Increase verbosity")

    args = parser.parse_args()
    test(args)
