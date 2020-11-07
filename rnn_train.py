import argparse
import math
import os
import pickle
import time

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from models import LSTMClassifier
from rnn_features import DATA_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i

def repackage_hidden(h):
    """Detach hidden states from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def make_batches(features, labels, batch_size):
    # Pad sequences so they are of equal length
    features = pad_sequence(features).T
    # Calculate number of batches from batch size
    nbatch = len(features) // batch_size
    # Trim of data that doesn't fit
    features = features.narrow(0, 0, nbatch * batch_size)
    labels = torch.tensor(labels).narrow(0, 0, nbatch * batch_size)
    # Transform data into shape (num_batches, batch_size)
    features = features.reshape(-1, features.shape[1], batch_size)
    # TODO: make sure labels are still aligned with features
    labels = labels.reshape(-1, batch_size)
    return features.to(device), labels.to(device)

def train(args):
    print("Start training with args: ", args)
    print("Device: ", device)
    # Load data
    vocab = pickle.load(open(DATA_PATH + "vocab.p", "rb"))
    label_vocab = pickle.load(open(DATA_PATH + "vocab_labels.p", "rb"))

    train_features = pickle.load(open(DATA_PATH + "features_train.p", "rb"))
    train_labels = pickle.load(open(DATA_PATH + "labels_train.p", "rb"))
    train_features, train_labels = make_batches(train_features, train_labels, args.batch_size)

    val_features = pickle.load(open(DATA_PATH + "features_val.p", "rb"))
    val_labels = pickle.load(open(DATA_PATH + "labels_val.p", "rb"))
    val_features, val_labels = make_batches(val_features, val_labels, args.eval_batch_size)

    test_features = pickle.load(open(DATA_PATH + "features_test.p", "rb"))
    test_labels = pickle.load(open(DATA_PATH + "labels_test.p", "rb"))

    print("Loaded data.")

    model = LSTMClassifier(
        len(vocab),
        args.emsize,
        args.nhid,
        args.nlayers,
        args.dropout,
        len(label_vocab)
    ).to(device)

    criterion = nn.NLLLoss()

    def train_epoch(epoch, lr):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        hidden = model.init_hidden(args.batch_size)
        for batch, (samples, true_labels) in enumerate(zip(train_features, train_labels)):
            model.zero_grad()
            hidden = repackage_hidden(hidden)
            output, hidden = model(samples, hidden)
            loss = criterion(output[-1], true_labels)
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            # Update parameter weights
            for p in model.parameters():
                p.data.add_(p.grad.data, alpha=-lr)

            total_loss += loss.item()

            if batch % args.log_interval == 0 and batch > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                    "loss {:5.2f}".format(
                        epoch,
                        batch,
                        len(train_features),
                        lr,
                        elapsed * 1000 / args.log_interval,
                        cur_loss,
                    )
                )
                total_loss = 0
                start_time = time.time()
            if args.dry_run:
                break

    def evaluate(features, labels):
        # Turn on evaluation mode which disables dropout.
        model.eval()
        total_loss = 0.0
        hidden = model.init_hidden(args.eval_batch_size)
        with torch.no_grad():
            for samples, true_labels in zip(features, labels):
                output, hidden = model(samples, hidden)
                hidden = repackage_hidden(hidden)
                total_loss += criterion(output[-1], true_labels).item()
        return total_loss / (len(features) - 1)

    # Loop over epochs.
    lr = args.lr
    best_val_loss = None

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_epoch(epoch, lr)
            val_loss = evaluate(val_features, val_labels)
            print("-" * 89)
            print(
                "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
                "valid ppl {:8.2f}".format(
                    epoch,
                    (time.time() - epoch_start_time),
                    val_loss,
                    math.exp(val_loss),
                )
            )
            print("-" * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, "wb") as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")

    # Load the best saved model.
    with open(args.save, "rb") as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_labels, test_features)
    print("=" * 89)
    print(
        "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
            test_loss, math.exp(test_loss)
        )
    )
    print("=" * 89)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="./data/",
        help="location of the data corpus",
    )
    parser.add_argument(
        "--emsize", type=int, default=300, help="size of word embeddings"
    )
    parser.add_argument(
        "--nhid", type=int, default=200, help="number of hidden units per layer"
    )
    parser.add_argument("--nlayers", type=int, default=2, help="number of layers")
    # TODO: use Adam
    parser.add_argument("--lr", type=float, default=20, help="initial learning rate")
    parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=50, help="upper epoch limit")
    parser.add_argument(
        "--batch-size", type=int, default=50, metavar="N", help="batch size"
    )
    parser.add_argument(
        "--eval-batch_size",
        type=int,
        default=10,
        metavar="N",
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=20, help="sequence length"
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
