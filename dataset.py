import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def pad_batch(batch):
    # Pad sequences within a batch so they all have equal length
    # padded = pad_sequence([torch.LongTensor(s) for s, _, _, _, _ , _ in batch], batch_first=True)
    # context_length = len(batch[0][1])
    # padded_contexts = []
    # lengths_contexts = []
    # for i in range(context_length):
    #     padded_contexts.append(pad_sequence([torch.tensor(c[i]) for _, c, _, _, _, _ in batch], batch_first=True))
    #     lengths_contexts.append(torch.tensor([lc[i] for _, _, _, _, lc, _ in batch]))

    # targets = torch.tensor([t for _, _, t, _, _, _ in batch])
    # lengths = torch.tensor([l for _, _, _, l, _, _ in batch])
    # ages = torch.tensor([a for _, _, _, _, _, a in batch])


    # return padded, padded_contexts, targets, lengths, lengths_contexts, ages

    return batch


class SpeechActsDataset(Dataset):

    def __init__(self, dataframe):
        self.len = len(dataframe)
        self.data = dataframe

    def __getitem__(self, index):
        utterances = self.data.utterances[index]
        labels = self.data.labels[index]
        sequence_length = len(utterances)
        age = self.data.age_months[index]

        return utterances, labels, sequence_length, age

    def __len__(self):
        return self.len



