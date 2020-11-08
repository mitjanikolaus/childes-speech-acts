import argparse
import pickle

import torch
from torch.nn.utils.rnn import pad_sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_words(indices, vocab):
    return " ".join([vocab.itos[i] for i in indices])


def test(args):
    print("Start training with args: ", args)
    print("Device: ", device)
    # Load data
    vocab = pickle.load(open(args.data + "vocab.p", "rb"))
    label_vocab = pickle.load(open(args.data + "vocab_labels.p", "rb"))

    print("Loading data..")
    test_features = pickle.load(open(args.data + "features_test.p", "rb"))
    test_labels = pickle.load(open(args.data + "labels_test.p", "rb"))
    dataset_test = list(zip(test_features, test_labels))
    print("Test samples: ", len(dataset_test))

    def evaluate(dataset):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        num_total = 0
        num_correct = 0
        hidden = model.init_hidden(args.batch_size)
        with torch.no_grad():
            num_batches = len(dataset) // args.batch_size
            for batch_id in range(num_batches):
                # TODO last small batch is lost at the moment
                batch = dataset[
                    batch_id * args.batch_size : (batch_id + 1) * args.batch_size
                ]
                batch.sort(key=lambda x: len(x[0]), reverse=True)

                samples = [sample for sample, _ in batch]
                labels = torch.tensor([label for _, label in batch]).to(device)

                sequence_lengths = [len(sample) for sample in samples]
                padded_samples = pad_sequence(samples).to(device)

                output, hidden = model(padded_samples, hidden, sequence_lengths)

                # Take last output for each sample (which depends on the sequence length)
                indices = [s - 1 for s in sequence_lengths]
                output = output[indices, range(args.batch_size)]

                predicted_labels = torch.argmax(output, dim=1)

                for sample, label, predicted in zip(samples, labels, predicted_labels):
                    print(
                        f"{get_words(sample, vocab)} Predicted: {label_vocab.inverse[int(predicted)]} True: {label_vocab.inverse[int(label)]}"
                    )

                num_correct += int(torch.sum(predicted_labels == labels))
                num_total += len(batch)

        return num_correct / num_total

    # Load the saved model checkpoint.
    with open(args.checkpoint, "rb") as f:
        model = torch.load(f, map_location=device)
        model.lstm.flatten_parameters()

    # Run on test data.
    test_accuracy = evaluate(dataset_test)
    print("=" * 89)
    print("Test acc {:5.2f}".format(test_accuracy))
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
