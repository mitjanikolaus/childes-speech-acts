import argparse
import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader

from torch import cuda

from dataset import SpeechActsDataset
from models import SpeechActDistilBERT

device = "cuda" if cuda.is_available() else "cpu"


def calcuate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct


def train(args):
    vocab = pickle.load(open(args.data + "vocab.p", "rb"))
    label_vocab = pickle.load(open(args.data + "vocab_labels.p", "rb"))

    # TODO use BERT tokenizer?
    train_dataset = pd.read_hdf(args.data + "speech_acts_data.h5", "train")

    test_dataset = pd.read_hdf(args.data + "speech_acts_data.h5", "test")

    training_set = SpeechActsDataset(train_dataset)
    testing_set = SpeechActsDataset(test_dataset)

    train_params = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 0}

    test_params = {"batch_size": args.batch_size, "shuffle": True, "num_workers": 0}

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    model = SpeechActDistilBERT(num_classes=len(label_vocab), dropout=args.dropout)
    model.to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    def train_epoch(epoch):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        model.train()
        for _, (features, labels, attention_masks) in enumerate(training_loader, 0):
            features = features.to(device)
            labels = labels.to(device)
            attention_masks = attention_masks.to(device)

            # TODO pack sequences?
            outputs = model(features, attention_masks)

            # TODO take into account different sequence lengths?
            loss = loss_function(outputs, labels)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, labels)

            nb_tr_steps += 1
            nb_tr_examples += labels.size(0)

            if _ % 5000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                print(f"Training Loss per 5000 steps: {loss_step}")
                print(f"Training Accuracy per 5000 steps: {accu_step}")

            optimizer.zero_grad()
            loss.backward()
            # # When using GPU
            optimizer.step()

        print(
            f"The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}"
        )
        epoch_loss = tr_loss / nb_tr_steps
        epoch_accu = (n_correct * 100) / nb_tr_examples
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Training Accuracy Epoch: {epoch_accu}")

        return

    for epoch in range(args.epochs):
        train_epoch(epoch)

        # Saving the files for re-use
        # TODO: only if best model so far
        torch.save(model, args.save)
        print("Model checkpoint saved")

    def valid(model, testing_loader):
        model.eval()
        n_correct = 0
        n_wrong = 0
        total = 0
        tr_loss = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                ids = data["ids"].to(device, dtype=torch.long)
                mask = data["mask"].to(device, dtype=torch.long)
                targets = data["targets"].to(device, dtype=torch.long)
                outputs = model(ids, mask).squeeze()
                loss = loss_function(outputs, targets)
                tr_loss += loss.item()
                big_val, big_idx = torch.max(outputs.data, dim=1)
                n_correct += calcuate_accu(big_idx, targets)

                nb_tr_steps += 1
                nb_tr_examples += targets.size(0)

                if _ % 5000 == 0:
                    loss_step = tr_loss / nb_tr_steps
                    accu_step = (n_correct * 100) / nb_tr_examples
                    print(f"Validation Loss per 100 steps: {loss_step}")
                    print(f"Validation Accuracy per 100 steps: {accu_step}")
        epoch_loss = tr_loss / nb_tr_steps
        epoch_accu = (n_correct * 100) / nb_tr_examples
        print(f"Validation Loss Epoch: {epoch_loss}")
        print(f"Validation Accuracy Epoch: {epoch_accu}")

        return epoch_accu

    print(
        "This is the validation section to print the accuracy and see how it performs"
    )
    print(
        "Here we are leveraging on the dataloader crearted for the validation dataset, the approcah is using more of pytorch"
    )

    acc = valid(model, testing_loader)
    print("Accuracy on test data = %0.2f%%" % acc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="./data/",
        help="location of the data corpus and vocabs",
    )
    parser.add_argument("--lr", type=float, default=1e-05, help="initial learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="upper epoch limit")
    parser.add_argument(
        "--batch-size", type=int, default=50, metavar="N", help="batch size"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="dropout applied to layers (0 = no dropout)",
    )
    parser.add_argument("--seed", type=int, default=1111, help="random seed")
    parser.add_argument(
        "--save", type=str, default="model.pt", help="path to save the final model"
    )

    args = parser.parse_args()
    train(args)
