import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def pad_batch(batch):
    # Pad sequences within a batch so they all have equal length
    padded = pad_sequence([torch.LongTensor(s) for s, _, _, _, _ , _, _, _ in batch], batch_first=True)
    context_length = len(batch[0][1])
    padded_contexts = []
    lengths_contexts = []
    for i in range(context_length):
        padded_contexts.append(pad_sequence([torch.tensor(c[i]) for _, c, _, _, _, _, _, _ in batch], batch_first=True))
        lengths_contexts.append(torch.tensor([lc[i] for _, _, _, _, lc, _, _, _ in batch]))

    targets = torch.tensor([t for _, _, t, _, _, _, _, _ in batch])
    lengths = torch.tensor([l for _, _, _, l, _, _, _, _ in batch])

    padded_action = pad_sequence([torch.LongTensor(a) for _, _, _, _, _, a, _, _ in batch], batch_first=True)
    action_lengths = torch.tensor([l for _, _, _, _, _, _, l, _ in batch])

    ages = torch.tensor([a for _, _, _, _, _, _, _, a in batch])

    return padded, padded_contexts, targets, lengths, lengths_contexts, padded_action, action_lengths, ages

class SpeechActsDataset(Dataset):

    def __init__(self, dataframe, context=None, context_length=0):
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
                utt = self.context.utterance[global_index - i - 1]
                context.append(utt)
                sequence_lengths_context.append(len(utt))
            except KeyError:
                # continue
                #TODO fix: this is not the correct context
                utt = self.context.utterance[0]
                context.append(utt)
                sequence_lengths_context.append(len(utt))

        utterance = self.data.utterance[index]
        label = self.data.label[index]
        sequence_length = len(utterance)
        action = self.data.action[index]
        action_length = len(action)
        age = self.data.age[index]

        return utterance, context, label, sequence_length, sequence_lengths_context, action, action_length, age

    def __len__(self):
        return self.len



