import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class SpeechActsDataset(Dataset):

    def __init__(self, dataframe):
        self.sequence_lengths = [len(sample) for sample in dataframe["features"]]
        dataframe["features"] = pad_sequence([torch.LongTensor(f) for f in dataframe["features"]], batch_first=True)
        self.len = len(dataframe)
        self.data = dataframe
        self.max_len = len(dataframe["features"][0])

    def __getitem__(self, index):
        features = self.data["features"][index]
        label = self.data["labels"][index]

        sequence_length = self.sequence_lengths[index]
        attention_mask = sequence_length * [1] + (self.max_len - sequence_length) * [0]

        return torch.tensor(features), torch.tensor(label), torch.tensor(attention_mask)

    def __len__(self):
        return self.len



