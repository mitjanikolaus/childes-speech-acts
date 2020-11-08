import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.rnn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMClassifier(nn.Module):
    """LSTM Language Model."""

    def __init__(
        self, vocab_size, n_input_layer_units, n_hidden_units, n_layers, dropout, label_size
    ):
        super(LSTMClassifier, self).__init__()
        self.ntoken = vocab_size
        self.drop = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(vocab_size, n_input_layer_units)
        self.lstm = LSTM(n_input_layer_units, n_hidden_units, n_layers, dropout=dropout)

        self.decoder = nn.Linear(n_hidden_units, label_size)
        self.softmax = nn.LogSoftmax(dim=2)

        self.nhid = n_hidden_units
        self.nlayers = n_layers

    def forward(self, input: Tensor, hidden, sequence_lengths):
        # Expected input dimensions: (sequence_length, batch_size, number_of_features)
        emb = self.embeddings(input)

        packed_emb = pack_padded_sequence(emb, sequence_lengths)
        output, hidden = self.lstm(packed_emb, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)
        output = self.drop(output)
        decoded = self.decoder(output)
        return self.softmax(decoded), hidden

    def init_hidden(self, batch_size):
        parameters_input = next(self.parameters())
        return (
            parameters_input.new_zeros(self.nlayers, batch_size, self.nhid),
            parameters_input.new_zeros(self.nlayers, batch_size, self.nhid),
        )
