"""Training routine for LSTM and Transformer"""

import argparse
import os
import pickle

import pandas as pd

import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader

from nn_dataset import SpeechActsDataset
from nn_models import SpeechActLSTM, SpeechActDistilBERT
from preprocess import SPEECH_ACT
from utils import build_vocabulary, dataset_labels, preprend_speaker_token, get_words, TRAIN_TEST_SPLIT_RANDOM_STATE, \
    make_train_test_splits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_TRANSFORMER = "transformer"
MODEL_LSTM = "lstm"

VAL_SPLIT_SIZE = .1

def prepare_data(data, vocab, label_vocab):
    # Prepend speaker tokens
    data.tokens = data.apply(lambda row: preprend_speaker_token(row.tokens, row.speaker), axis=1)

    # Convert words and labels to indices using the respective vocabs
    data["utterances"] = data.tokens.apply(lambda tokens: [vocab.stoi[t] for t in tokens])
    data["labels"] = data[SPEECH_ACT].apply(lambda l: label_vocab[l])

    # Group by transcript (file name), each transcript is treated as one long input sequence
    data_grouped = data.groupby(by=['file_id']).agg({
        'utterances': lambda x: [y for y in x],
        'labels': lambda x: [y for y in x],
        'age_months': min,
    })
    return data_grouped

def train(args):
    print("Start training with args: ", args)
    print("Device: ", device)

    # Load data
    data = pd.read_pickle(args.data)

    data_train, data_test = make_train_test_splits(data, args.test_ratio)

    print("Building vocabulary..")
    vocab = build_vocabulary(data_train["tokens"], args.vocab_size)
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    pickle.dump(vocab, open(os.path.join(args.out, "vocab.p"), "wb"))

    label_vocab = dataset_labels()
    pickle.dump(label_vocab, open(os.path.join(args.out, "vocab_labels.p"), "wb"))

    data_train = prepare_data(data_train, vocab, label_vocab)
    data_test = prepare_data(data_test, vocab, label_vocab)

    data_train, data_val = train_test_split(
        data_train, test_size=VAL_SPLIT_SIZE, shuffle=True, random_state=TRAIN_TEST_SPLIT_RANDOM_STATE
    )

    dataset_train = SpeechActsDataset(data_train)
    dataset_val = SpeechActsDataset(data_val)
    dataset_test = SpeechActsDataset(data_test)

    train_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    print("Loaded data.")

    if args.model == MODEL_LSTM:
        model = SpeechActLSTM(
            len(vocab), args.emsize, args.nhid_words_lstm, args.nhid_utterance_lstm, args.nlayers, args.dropout, len(label_vocab)
        )
    elif args.model == MODEL_TRANSFORMER:
        model = SpeechActDistilBERT(len(label_vocab), args.dropout)
    else:
        raise RuntimeError("Unknown model type: ", args.model)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    def train_epoch(data_loader, epoch):
        model.train()
        total_loss = 0.0

        for batch_id, (input_samples, targets, sequence_lengths, ages) in enumerate(data_loader):
            # Move data to GPU
            targets = torch.tensor(targets).to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Perform forward pass of the model
            loss = model(input_samples, targets)

            # Calculate loss
            # loss = criterion(output, targets)
            total_loss += loss.item()
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            # Update parameter weights
            optimizer.step()

            if batch_id % args.log_interval == 0 and batch_id != 0:
                cur_loss = total_loss / (args.log_interval * args.batch_size)
                current_learning_rate = optimizer.param_groups[0]["lr"]
                print(
                    "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | loss {:5.5f}".format(
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
        with torch.no_grad():
            for batch_id, (input_samples, targets, sequence_lengths, ages) in enumerate(data_loader):
                # Move data to GPU
                targets = torch.tensor(targets).to(device)

                # Perform forward pass of the model
                predicted_labels = model.forward_decode(input_samples)
                predicted_labels = torch.tensor(predicted_labels).to(device)
                # Calculate loss
                # loss = criterion(output, targets)
                # total_loss += loss.item()

                # Compare predicted labels to ground truth
                # predicted_labels = torch.argmax(output, dim=1)
                num_correct += int(torch.sum(predicted_labels == targets))
                num_samples += len(input_samples)

        return total_loss / num_samples, num_correct / num_samples

    # Loop over epochs.
    best_val_acc = None

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
            if not best_val_acc or val_accuracy > best_val_acc:
                with open(os.path.join(args.out, "model.pt"), "wb") as f:
                    torch.save(model, f)
                best_val_acc = val_accuracy

    except KeyboardInterrupt:
        print("-" * 89)
        print("Exiting from training early")

    # Load the best saved model.
    with open(f"{args.out}/model.pt", "rb") as f:
        model = torch.load(f)

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
        required=True,
        help="path to the data corpus",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/",
        help="directory to store result files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_TRANSFORMER,
        choices=[MODEL_TRANSFORMER, MODEL_LSTM],
        help="model architecture",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Ratio of dataset to be used to testing",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=1000,
        help="Maxmimum size of the vocabulary",
    )
    parser.add_argument(
        "--emsize", type=int, default=200, help="size of word embeddings"
    )
    parser.add_argument(
        "--nhid-words-lstm", type=int, default=200, help="number of hidden units of the lower-level LSTM"
    )
    parser.add_argument(
        "--nhid-utterance-lstm", type=int, default=100, help="number of hidden units of the higher-level LSTM"
    )

    parser.add_argument("--nlayers", type=int, default=2, help="number of layers of the lower-level LSTM")
    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--clip", type=float, default=0.25, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=50, help="upper epoch limit")
    # TODO fix: works only with batch size one at the moment
    parser.add_argument(
        "--batch-size", type=int, default=1, metavar="N", help="batch size"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="dropout applied to layers (0 = no dropout)",
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument(
        "--log-interval", type=int, default=30, metavar="N", help="report interval"
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="verify the code and the model"
    )

    args = parser.parse_args()
    train(args)
