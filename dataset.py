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

    def __init__(self, dataframe, context=None, context_length=0):
        self.sequence_lengths = [len(sample) for sample in dataframe["features"]]
        self.len = len(dataframe)
        self.data = dataframe

        # Data to to take context from
        self.context = context if context is not None else dataframe

        # Number of previous utterances that are provided as features
        self.context_length = context_length

    def __getitem__(self, index):
        # global index to address context
        global_index = index
        if not len(self.data) == len(self.context):
            global_index = self.data.global_index[index]

        # Add context to features
        features = []
        for i in reversed(range(self.context_length)):
            try:
                features += self.context.features[global_index - i - 1]
            except KeyError:
                continue

        # Add words of target utterance
        features += self.data.features[index]

        label = self.data.labels[index]

        sequence_length = self.sequence_lengths[index]

        return torch.tensor(features), torch.tensor(label), torch.tensor(sequence_length)

    def __len__(self):
        return self.len



