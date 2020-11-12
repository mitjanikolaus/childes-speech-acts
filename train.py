"""Training routine for LSTM and Transformer"""

import argparse
import pickle

import pandas as pd

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import SpeechActsDataset, pad_batch
from models import SpeechActLSTM, SpeechActDistilBERT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_TRANSFORMER = "transformer"
MODEL_LSTM = "lstm"

def detach_hidden(h):
    """Detach hidden states from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(detach_hidden(v) for v in h)


def train(args):
    print("Start training with args: ", args)
    print("Device: ", device)
    # Load data
    vocab = pickle.load(open(args.data + "vocab.p", "rb"))
    label_vocab = pickle.load(open(args.data + "vocab_labels.p", "rb"))

    train_dataframe = pd.read_hdf(args.data + "speech_acts_data.h5", "train")
    val_dataframe = pd.read_hdf(args.data + "speech_acts_data.h5", "val")
    test_dataframe = pd.read_hdf(args.data + "speech_acts_data.h5", "test")

    dataset_train = SpeechActsDataset(train_dataframe)
    dataset_val = SpeechActsDataset(val_dataframe)
    dataset_test = SpeechActsDataset(test_dataframe)

    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_batch,
    )
    valid_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_batch,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_batch,
    )
    print("Loaded data.")

    if args.model == MODEL_LSTM:
        model = SpeechActLSTM(
            len(vocab), args.emsize, args.nhid, args.nlayers, args.dropout, len(label_vocab)
        )
    elif args.model == MODEL_TRANSFORMER:
        model = SpeechActDistilBERT(num_classes=len(label_vocab), dropout=args.dropout)
    else:
        raise RuntimeError("Unknown model type: ",args.model)

    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def train_epoch(data_loader, epoch):
        model.train()
        total_loss = 0.0
        if args.model == MODEL_LSTM:
            hidden = model.init_hidden(args.batch_size)

        for batch_id, (input_samples, targets, sequence_lengths) in enumerate(data_loader):
            # Move data to GPU
            input_samples = input_samples.to(device)
            targets = targets.to(device)
            sequence_lengths = sequence_lengths.to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Perform forward pass of the model
            if args.model == MODEL_LSTM:
                output, hidden = model(input_samples, hidden, sequence_lengths)
            else:
                output = model(input_samples, sequence_lengths)

            # Calculate loss
            loss = criterion(output, targets)
            total_loss += loss.item()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            # Update parameter weights
            optimizer.step()

            if args.model == MODEL_LSTM:
                hidden = detach_hidden(hidden)

            if batch_id % args.log_interval == 0:
                cur_loss = total_loss / (args.log_interval * args.batch_size)
                current_learning_rate = optimizer.param_groups[0]["lr"]
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | loss {:5.5f}".format(
                        epoch,
                        batch_id,
                        len(data_loader),
                        current_learning_rate,
                        cur_loss,
                    )
                )
                total_loss = 0

            if args.dry_run:
                break

    def evaluate(data_loader):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.0
        num_samples = 0
        num_correct = 0
        if args.model == MODEL_LSTM:
            hidden = model.init_hidden(args.batch_size)
        with torch.no_grad():
            for batch_id, (input_samples, targets, sequence_lengths) in enumerate(
                data_loader
            ):
                # Move data to GPU
                input_samples = input_samples.to(device)
                targets = targets.to(device)
                sequence_lengths = sequence_lengths.to(device)

                # Clear gradients
                optimizer.zero_grad()

                # Perform forward pass of the model
                if args.model == MODEL_LSTM:
                    output, hidden = model(input_samples, hidden, sequence_lengths)
                else:
                    output = model(input_samples, sequence_lengths)

                # Calculate loss
                loss = criterion(output, targets)
                total_loss += loss.item()

                # Compare predicted labels to ground truth
                predicted_labels = torch.argmax(output, dim=1)
                num_correct += int(torch.sum(predicted_labels == targets))
                num_samples += len(input_samples)

        return total_loss / len(data_loader), num_correct / num_samples

    # Loop over epochs.
    best_val_loss = None

    try:
        for epoch in range(1, args.epochs + 1):
            train_epoch(train_loader, epoch)
            val_loss, val_accuracy = evaluate(valid_loader)
            print("-" * 89)
            print(
                "| end of epoch {:3d} | valid loss {:5.5f} | valid acc {:5.2f} ".format(
                    epoch, val_loss, val_accuracy
                )
            )
            print("-" * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, "wb") as f:
                    torch.save(model, f)
                best_val_loss = val_loss

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")

    # Load the best saved model.
    with open(args.save, "rb") as f:
        model = torch.load(f)
        if args.model == MODEL_LSTM:
            model.lstm.flatten_parameters()

    # Run on test data.
    test_loss, test_accuracy = evaluate(test_loader)
    print("=" * 89)
    print(
        "| End of training | test loss {:5.2f} | test acc {:5.2f}".format(
            test_loss, test_accuracy
        )
    )
    print("=" * 89)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="./data/",
        help="location of the data corpus and vocabs",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_TRANSFORMER,
        choices=[MODEL_TRANSFORMER, MODEL_LSTM],
        help="model architecture",
    )
    parser.add_argument(
        "--emsize", type=int, default=300, help="size of word embeddings"
    )
    parser.add_argument(
        "--nhid", type=int, default=200, help="number of hidden units per layer"
    )
    parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=20, help="upper epoch limit")
    parser.add_argument(
        "--batch-size", type=int, default=50, metavar="N", help="batch size"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="dropout applied to layers (0 = no dropout)",
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument(
        "--log-interval", type=int, default=200, metavar="N", help="report interval"
    )
    parser.add_argument(
        "--save", type=str, default="model.pt", help="path to save the final model"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="verify the code and the model"
    )

    args = parser.parse_args()
    train(args)
