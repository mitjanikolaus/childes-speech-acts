import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def pad_batch(batch):
    # Pad sequences within a batch so they all have equal length
    padded = pad_sequence([torch.LongTensor(s) for s, _, _, _, _ in batch], batch_first=True)
    context_length = len(batch[0][1])
    padded_contexts = []
    lengths_contexts = []
    for i in range(context_length):
        padded_contexts.append(pad_sequence([torch.tensor(c[i]) for _, c, _, _, _ in batch], batch_first=True))
        lengths_contexts.append(torch.tensor([lc[i] for _, _, _, _, lc in batch]))

    targets = torch.tensor([t for _, _, t, _, _ in batch])
    lengths = torch.tensor([l for _, _, _, l, _ in batch])

    return padded, padded_contexts, targets, lengths, lengths_contexts

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
        sequence_lengths_context = []
        for i in reversed(range(self.context_length)):
            try:
                utt = self.context.features[global_index - i - 1]
                context.append(utt)
                sequence_lengths_context.append(len(utt))
            except KeyError:
                # continue
                #TODO fix: this is not the correct context
                utt = self.context.features[0]
                context.append(utt)
                sequence_lengths_context.append(len(utt))



        features = self.data.features[index]

        label = self.data.labels[index]

        # TODO sequence lengths can be accessed directly!
        sequence_length = self.sequence_lengths[index]

        return torch.tensor(features), context, torch.tensor(label), torch.tensor(sequence_length), torch.tensor(sequence_lengths_context)

    def __len__(self):
        return self.len



