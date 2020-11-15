import argparse
import pickle

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from torch.utils.data import DataLoader

from dataset import SpeechActsDataset, pad_batch
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
    data = pd.read_hdf(args.data + "speech_acts_data.h5", "test")

    # Separate data by speaker
    data.reset_index(inplace=True)
    data_adults = data[data.utterance.apply(lambda x: x[0] == vocab.stoi[SPEAKER_ADULT])]
    data_children = data[data.utterance.apply(lambda x: x[0] == vocab.stoi[SPEAKER_CHILD])]

    data_children["global_index"] = data_children["index"].copy()
    data_adults["global_index"] = data_adults["index"].copy()
    data_adults.reset_index(inplace=True)
    data_children.reset_index(inplace=True)

    dataset_adults = SpeechActsDataset(data_adults, context=data, context_length=args.context)
    dataset_children = SpeechActsDataset(data_children, context=data, context_length=args.context)
    loader_adults = DataLoader(
        dataset_adults,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_batch,
    )
    loader_children = DataLoader(
        dataset_children,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_batch,
    )
    print("Test samples (adult): ", len(data_adults))
    print("Test samples (children): ", len(data_children))

    def evaluate(data_loader):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        num_total = 0
        num_correct = 0
        all_true_labels = []
        all_predicted_labels = []
        all_ages = []
        with torch.no_grad():
            for batch_id, (
                    input_samples, input_contexts, targets, sequence_lengths, sequence_lengths_context, ages) in enumerate(
                data_loader):
                # Move data to GPU
                input_samples = input_samples.to(device)
                # input_contexts = input_contexts.to(device)
                targets = targets.to(device)
                sequence_lengths = sequence_lengths.to(device)
                # sequence_lengths_context = sequence_lengths_context.to(device)

                # Perform forward pass of the model
                output = model(input_samples, input_contexts, sequence_lengths, sequence_lengths_context)

                # Compare predicted labels to ground truth
                predicted_labels = torch.argmax(output, dim=1)
                num_correct += int(torch.sum(predicted_labels == targets))
                num_total += len(input_samples)

                predicted_labels = torch.argmax(output, dim=1)

                all_true_labels += targets.tolist()
                all_predicted_labels += predicted_labels.tolist()
                all_ages += ages.tolist()

                if args.verbose:
                    for i, (sample, label, predicted) in enumerate(zip(input_samples, targets, predicted_labels)):
                        if label_vocab.inverse[int(predicted)] != label_vocab.inverse[int(label)]:
                            for j in range(args.context):
                                print(get_words(input_contexts[j][i], vocab), end="")
                            print(
                                f"{get_words(sample, vocab)} Predicted: {label_vocab.inverse[int(predicted)]} True: {label_vocab.inverse[int(label)]}"
                            )

                num_correct += int(torch.sum(predicted_labels == targets))
                num_total += len(input_samples)

        test_accuracy = num_correct / num_total
        print("=" * 89)
        print("Test acc: {:5.2f}".format(test_accuracy))
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

        return test_accuracy

    # Load the saved model checkpoint.
    with open(args.checkpoint, "rb") as f:
        model = torch.load(f, map_location=device)
        if isinstance(model, SpeechActLSTM):
            model.lstm.flatten_parameters()

    # Run on test data.
    print("Eval (adults' utterances)")
    evaluate(loader_adults)

    print("Eval (children's utterances)")
    evaluate(loader_children)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
        help="location of the data corpus and vocabs",
    )
    parser.add_argument(
        "--context", type=int, default=0, help="Number of previous utterances that are provided as features"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, metavar="N", help="batch size"
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
