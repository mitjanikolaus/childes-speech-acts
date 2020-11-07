import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.rnn import LSTM


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

        self.init_weights()

        self.nhid = n_hidden_units
        self.nlayers = n_layers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.embeddings.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input: Tensor, hidden):
        # Expected input dimensions: (sequence_length, batch_size, number_of_features)
        emb = self.embeddings(input)
        output, hidden = self.lstm(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        # decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden #, dim=1), hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.nlayers, batch_size, self.nhid),
            weight.new_zeros(self.nlayers, batch_size, self.nhid),
        )
