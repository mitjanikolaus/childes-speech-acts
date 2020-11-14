import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def pad_batch(batch):
    # Pad sequences within a batch so they all have equal length
    padded = pad_sequence([torch.LongTensor(s) for s, _, _, _, _ in batch], batch_first=True)
    padded_contexts = pad_sequence([torch.LongTensor(c) for _, c, _, _, _ in batch], batch_first=True)
    targets = torch.tensor([t for _, _, t, _, _ in batch])
    lengths = torch.tensor([l for _, _, _, l, _ in batch])
    lengths_context = torch.tensor([lc for _, _, _, _, lc in batch])

    return padded, padded_contexts, targets, lengths, lengths_context

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

        # context
        context = []
        sequence_length_context = 0
        for i in reversed(range(self.context_length)):
            try:
                context += self.context.features[global_index - i - 1]
                sequence_length_context += len(context)
            except KeyError:
                # continue
                #TODO fix: this is not the correct context
                context += self.context.features[0]
                sequence_length_context += len(context)



        features = self.data.features[index]

        label = self.data.labels[index]

        # TODO sequence lengths can be accessed directly!
        sequence_length = self.sequence_lengths[index]

        return torch.tensor(features), torch.tensor(context), torch.tensor(label), torch.tensor(sequence_length), torch.tensor(sequence_length_context)

    def __len__(self):
        return self.len



