import argparse
import pickle

import torch
import pandas as pd
from torch.utils.data import DataLoader

from dataset import SpeechActsDataset, pad_batch
from generate_dataset import PADDING, SPEAKER_ADULT, SPEAKER_CHILD
from models import SpeechActDistilBERT, SpeechActLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_words(indices, vocab):
    return " ".join([vocab.itos[i] for i in indices if not vocab.itos[i] == PADDING])


def test(args):
    print("Start training with args: ", args)
    print("Device: ", device)
    # Load data
    vocab = pickle.load(open(args.data + "vocab.p", "rb"))
    label_vocab = pickle.load(open(args.data + "vocab_labels.p", "rb"))

    print("Loading data..")
    test_dataframe = pd.read_hdf(args.data + "speech_acts_data.h5", "test")

    # Separate data by speaker
    data_adults = test_dataframe[test_dataframe.features.apply(lambda x: x[0] == vocab.stoi[SPEAKER_ADULT])]
    data_children = test_dataframe[test_dataframe.features.apply(lambda x: x[0] == vocab.stoi[SPEAKER_CHILD])]

    data_adults.reset_index(drop=True, inplace=True)
    data_children.reset_index(drop=True, inplace=True)

    dataset_adults = SpeechActsDataset(data_adults)
    dataset_children = SpeechActsDataset(data_children)
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
    print("Test samples (adult): ", len(data_children))

    def evaluate(data_loader):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        num_total = 0
        num_correct = 0
        if isinstance(model, SpeechActLSTM):
            hidden = model.init_hidden(args.batch_size)
        with torch.no_grad():
            for batch_id, (input_samples, targets, sequence_lengths) in enumerate(
                    data_loader
            ):
                input_samples = input_samples.to(device)
                targets = targets.to(device)
                sequence_lengths = sequence_lengths.to(device)

                if isinstance(model, SpeechActLSTM):
                    output, hidden = model(input_samples, hidden, sequence_lengths)
                else:
                    output = model(input_samples, sequence_lengths)

                predicted_labels = torch.argmax(output, dim=1)

                for sample, label, predicted in zip(input_samples, targets, predicted_labels):
                    print(
                        f"{get_words(sample, vocab)} Predicted: {label_vocab.inverse[int(predicted)]} True: {label_vocab.inverse[int(label)]}"
                    )

                num_correct += int(torch.sum(predicted_labels == targets))
                num_total += len(input_samples)

        return num_correct / num_total

    # Load the saved model checkpoint.
    with open(args.checkpoint, "rb") as f:
        model = torch.load(f, map_location=device)
        if isinstance(model, SpeechActLSTM):
            model.lstm.flatten_parameters()

    # Run on test data.
    test_accuracy_adults = evaluate(loader_adults)
    test_accuracy_children = evaluate(loader_children)
    print("=" * 89)
    print("Test acc (adults' utterances): {:5.2f}".format(test_accuracy_adults))
    print("Test acc (children's utterances): {:5.2f}".format(test_accuracy_children))
    print("=" * 89)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/",
        help="location of the data corpus and vocabs",
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

    args = parser.parse_args()
    test(args)
