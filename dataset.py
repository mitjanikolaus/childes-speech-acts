import torch
from torch.utils.data import Dataset

class SpeechActsDataset(Dataset):

    def __init__(self, dataframe, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.max_len = max_len

    def __getitem__(self, index):
        features = self.data[index][0]
        label = self.data[index][1]

        return torch.tensor(features), torch.tensor(label)

    def __len__(self):
        return self.len



