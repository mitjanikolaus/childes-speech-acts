import torch
import torch.nn as nn
from torch import Tensor, cuda
from torch.nn.modules.rnn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torchcrf import CRF
from transformers import DistilBertModel

device = "cuda" if cuda.is_available() else "cpu"

class SpeechActLSTM(nn.Module):
    """LSTM Language Model."""

    def __init__(
        self,
        vocab_size,
        n_input_layer_units,
        n_hidden_units_words_lstm,
        n_hidden_units_utterance_lstm,
        n_layers_words_lstm,
        dropout,
        label_size,
    ):
        super(SpeechActLSTM, self).__init__()
        self.ntoken = vocab_size
        self.drop = nn.Dropout(dropout)
        self.embeddings = nn.Embedding(vocab_size, n_input_layer_units)
        self.lstm_words = LSTM(n_input_layer_units, n_hidden_units_words_lstm, n_layers_words_lstm, dropout=dropout)

        # self.lstm_utterance = LSTM(n_hidden_units_words_lstm, n_hidden_units_utterance_lstm, 1)

        self.decoder = nn.Linear(n_hidden_units_words_lstm, label_size)

        self.crf = CRF(label_size)

        self.n_hidden_units_utterance_lstm = n_hidden_units_utterance_lstm
        self.n_hidden_units_words_lstm = n_hidden_units_words_lstm
        self.n_layers_words_lstm = n_layers_words_lstm

    def forward(self, input: Tensor, targets: Tensor):
        # TODO use targets to train using teacher forcing

        sequence_lengths = [len(i) for i in input]
        padded_inputs = pad_sequence([torch.LongTensor(i).to(device) for i in input])

        emb = self.embeddings(padded_inputs)
        batch_size = emb.size(1)
        hidden = self.init_hidden(self.n_layers_words_lstm, batch_size, self.n_hidden_units_words_lstm)

        packed_emb = pack_padded_sequence(emb, sequence_lengths, enforce_sorted=False)
        output, hidden = self.lstm_words(packed_emb, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        # Take last output for each sample (which depends on the sequence length)
        indices = [s - 1 for s in sequence_lengths]
        utterance_representations = output[indices, range(batch_size)]

        decoded = self.decoder(utterance_representations)

        decoded = decoded.unsqueeze(1)
        targets = targets.unsqueeze(1)

        log_likelihood = self.crf(decoded, targets)

        return log_likelihood

        # hidden_utterance_lstm = self.init_hidden(1, 1, self.n_hidden_units_utterance_lstm)
        # outputs = []
        # for utt in utterance_representations:
        #     utt = utt.unsqueeze(0).unsqueeze(0) # add batch size and sequence length dimension
        #     output_utterance_level, hidden_utterance_lstm = self.lstm_utterance(utt, hidden_utterance_lstm)
        #
        #     outputs.append(self.decoder(output_utterance_level[0][0]))


        # return torch.stack(outputs)

    def forward_decode(self,  input: Tensor):
        # TODO use targets to train using teacher forcing

        sequence_lengths = [len(i) for i in input]
        padded_inputs = pad_sequence([torch.LongTensor(i).to(device) for i in input])

        emb = self.embeddings(padded_inputs)
        batch_size = emb.size(1)
        hidden = self.init_hidden(self.n_layers_words_lstm, batch_size, self.n_hidden_units_words_lstm)

        packed_emb = pack_padded_sequence(emb, sequence_lengths, enforce_sorted=False)
        output, hidden = self.lstm_words(packed_emb, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        # Take last output for each sample (which depends on the sequence length)
        indices = [s - 1 for s in sequence_lengths]
        utterance_representations = output[indices, range(batch_size)]

        decoded = self.decoder(utterance_representations)

        decoded = decoded.unsqueeze(1)

        labels = self.crf.decode(decoded)

        return labels[0]

    def init_hidden(self, n_layers, batch_size, n_hidden_units):
        parameters_input = next(self.parameters())
        return (
            parameters_input.new_zeros(n_layers, batch_size, n_hidden_units),
            parameters_input.new_zeros(n_layers, batch_size, n_hidden_units),
        )


class SpeechActDistilBERT(torch.nn.Module):

    def __init__(self, num_classes, dropout, finetune_bert=True):
        N_UNITS_BERT_OUT = 768

        super(SpeechActDistilBERT, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(N_UNITS_BERT_OUT, N_UNITS_BERT_OUT)
        self.dropout = torch.nn.Dropout(dropout)
        # self.classifier = torch.nn.Linear(N_UNITS_BERT_OUT*(1+context_length), num_classes)

        if not finetune_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def gen_attention_masks(self, sequence_lengths, max_len):
        return torch.tensor(
            [
                int(s) * [1] + (max_len - int(s)) * [0]
                for s in sequence_lengths
            ]
        ).to(device)

    def forward(self, input: Tensor, context: Tensor, sequence_lengths, sequence_lengths_context):
        # The input should be padded, so all samples should have the same length

        max_len = input[0].size(0)
        attention_masks = self.gen_attention_masks(sequence_lengths, max_len)
        output_1 = self.bert(input_ids=input, attention_mask=attention_masks)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)

        outputs = self.dropout(pooler)

        for context_utt, length in zip(context, sequence_lengths_context):
            context_utt = context_utt.to(device)
            length = length.to(device)
            
            max_len = context_utt[0].size(0)
            attention_masks = self.gen_attention_masks(length, max_len)
            output_1 = self.bert(input_ids=context_utt, attention_mask=attention_masks)
            hidden_state = output_1[0]
            pooler = hidden_state[:, 0]
            pooler = self.pre_classifier(pooler)
            pooler = torch.nn.ReLU()(pooler)
            outputs_context = self.dropout(pooler)

            # Append output
            outputs = torch.cat([outputs, outputs_context], dim=1)

        output = self.classifier(outputs)
        return output
