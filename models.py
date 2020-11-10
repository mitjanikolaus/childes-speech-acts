import torch
import torch.nn as nn
from torch import Tensor, cuda
from torch.nn.modules.rnn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import DistilBertModel

device = "cuda" if cuda.is_available() else "cpu"

class SpeechActLSTM(nn.Module):
    """LSTM Language Model."""

    def __init__(
        self,
        vocab_size,
        n_input_layer_units,
        n_hidden_units,
        n_layers,
        dropout,
        label_size,
    ):
        super(SpeechActLSTM, self).__init__()
        self.ntoken = vocab_size
        self.drop = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(vocab_size, n_input_layer_units)
        self.lstm = LSTM(n_input_layer_units, n_hidden_units, n_layers, dropout=dropout)

        self.decoder = nn.Linear(n_hidden_units, label_size)

        self.nhid = n_hidden_units
        self.nlayers = n_layers

    def forward(self, input: Tensor, hidden, sequence_lengths):
        # Expected input dimensions: (batch_size, sequence_length, number_of_features)
        emb = self.embeddings(input)

        packed_emb = pack_padded_sequence(emb, sequence_lengths, enforce_sorted=False, batch_first=True)
        output, hidden = self.lstm(packed_emb, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        output = self.drop(output)
        output = self.decoder(output)

        # Take last output for each sample (which depends on the sequence length)
        indices = [s - 1 for s in sequence_lengths]
        output = output[indices, range(len(input))]
        return output, hidden

    def init_hidden(self, batch_size):
        parameters_input = next(self.parameters())
        return (
            parameters_input.new_zeros(self.nlayers, batch_size, self.nhid),
            parameters_input.new_zeros(self.nlayers, batch_size, self.nhid),
        )


class SpeechActDistilBERT(torch.nn.Module):
    def __init__(self, num_classes, dropout):
        super(SpeechActDistilBERT, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # TODO optimize size
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(768, num_classes)

    def gen_attention_masks(self, sequence_lengths, max_len):
        return torch.tensor(
            [
                int(s) * [1] + (max_len - int(s)) * [0]
                for s in sequence_lengths
            ]
        ).to(device)

    def forward(self, input, sequence_lengths):
        # The input should be padded, so all samples should have the same length
        max_len = len(input[0])
        attention_masks = self.gen_attention_masks(sequence_lengths, max_len)
        output_1 = self.bert(input_ids=input, attention_mask=attention_masks)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
