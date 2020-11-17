import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset



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



