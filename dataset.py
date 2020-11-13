import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def pad_batch(batch):
    # Pad sequences within a batch so they all have equal length
    padded = pad_sequence([torch.LongTensor(s) for s, _, _ in batch], batch_first=True)
    targets = torch.tensor([t for _, t, _ in batch])
    lengths = torch.tensor([l for _, _, l in batch])

    return padded, targets, lengths

class SpeechActsDataset(Dataset):

    def __init__(self, dataframe, context_length=0):
        self.sequence_lengths = [len(sample) for sample in dataframe["features"]]
        self.len = len(dataframe)
        self.data = dataframe

        # Number of previous utterances that are provided as features
        self.context_length = context_length

    def __getitem__(self, index):
        features = []
        for i in reversed(range(self.context_length+1)):
            try:
                features += self.data.features[index - i]
            except KeyError:
                continue

        label = self.data.labels[index]

        sequence_length = self.sequence_lengths[index]

        return torch.tensor(features), torch.tensor(label), torch.tensor(sequence_length)

    def __len__(self):
        return self.len



