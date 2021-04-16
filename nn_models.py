import torch
import torch.nn as nn
from torch import cuda
from torch.nn.modules.rnn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torchcrf import CRF
from transformers import DistilBertModel

device = "cuda" if cuda.is_available() else "cpu"


class SpeechActLSTM(nn.Module):
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
        self.lstm_words = LSTM(
            n_input_layer_units,
            n_hidden_units_words_lstm,
            n_layers_words_lstm,
            dropout=dropout,
        )

        self.lstm_utterance = LSTM(
            n_hidden_units_words_lstm, n_hidden_units_utterance_lstm, 1
        )

        self.decoder = nn.Linear(n_hidden_units_utterance_lstm, label_size)

        self.crf = CRF(label_size)

        self.n_hidden_units_utterance_lstm = n_hidden_units_utterance_lstm
        self.n_hidden_units_words_lstm = n_hidden_units_words_lstm
        self.n_layers_words_lstm = n_layers_words_lstm

    def forward(self, input, targets):
        sequence_lengths = [len(i) for i in input]
        padded_inputs = pad_sequence([torch.LongTensor(i).to(device) for i in input])

        emb = self.embeddings(padded_inputs)
        batch_size = emb.size(1)
        hidden = self.init_hidden(
            self.n_layers_words_lstm, batch_size, self.n_hidden_units_words_lstm
        )

        packed_emb = pack_padded_sequence(emb, sequence_lengths, enforce_sorted=False)
        output, hidden = self.lstm_words(packed_emb, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        # Take last output for each sample (which depends on the sequence length)
        indices = [s - 1 for s in sequence_lengths]
        utterance_representations = output[indices, range(batch_size)]

        hidden_utterance_lstm = self.init_hidden(
            1, 1, self.n_hidden_units_utterance_lstm
        )

        utterance_representations = utterance_representations.unsqueeze(
            1
        )  # add batch size dimension
        output_utterance_level, hidden_utterance_lstm = self.lstm_utterance(
            utterance_representations, hidden_utterance_lstm
        )

        outputs = self.decoder(output_utterance_level.squeeze(1))

        decoded = outputs.unsqueeze(1)
        targets = targets.unsqueeze(1)

        log_likelihood = self.crf.forward(decoded, targets, reduction="token_mean")

        loss = -log_likelihood

        return loss

    def forward_decode(self, input):
        sequence_lengths = [len(i) for i in input]
        padded_inputs = pad_sequence([torch.LongTensor(i).to(device) for i in input])

        emb = self.embeddings(padded_inputs)
        batch_size = emb.size(1)
        hidden = self.init_hidden(
            self.n_layers_words_lstm, batch_size, self.n_hidden_units_words_lstm
        )

        packed_emb = pack_padded_sequence(emb, sequence_lengths, enforce_sorted=False)
        output, hidden = self.lstm_words(packed_emb, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        # Take last output for each sample (which depends on the sequence length)
        indices = [s - 1 for s in sequence_lengths]
        utterance_representations = output[indices, range(batch_size)]

        hidden_utterance_lstm = self.init_hidden(
            1, 1, self.n_hidden_units_utterance_lstm
        )

        utterance_representations = utterance_representations.unsqueeze(
            1
        )  # add batch size dimension
        output_utterance_level, hidden_utterance_lstm = self.lstm_utterance(
            utterance_representations, hidden_utterance_lstm
        )

        outputs = self.decoder(output_utterance_level.squeeze(1))

        decoded = outputs.unsqueeze(1)

        labels = self.crf.decode(decoded)

        return labels[0]

    def init_hidden(self, n_layers, batch_size, n_hidden_units):
        parameters_input = next(self.parameters())
        return (
            parameters_input.new_zeros(n_layers, batch_size, n_hidden_units),
            parameters_input.new_zeros(n_layers, batch_size, n_hidden_units),
        )


class SpeechActDistilBERT(torch.nn.Module):

    criterion = torch.nn.CrossEntropyLoss()

    def __init__(self, num_classes, dropout, finetune_bert=True):
        N_UNITS_BERT_OUT = 768

        super(SpeechActDistilBERT, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(N_UNITS_BERT_OUT, N_UNITS_BERT_OUT)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(N_UNITS_BERT_OUT, num_classes)

        if not finetune_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def gen_attention_masks(self, sequence_lengths, max_len):
        return torch.tensor(
            [int(s) * [1] + (max_len - int(s)) * [0] for s in sequence_lengths]
        ).to(device)

    def forward(self, input, targets):
        # The input should be padded, so all samples should have the same length
        sequence_lengths = [len(i) for i in input]
        padded_inputs = pad_sequence(
            [torch.LongTensor(i).to(device) for i in input], batch_first=True
        )

        max_len = max(sequence_lengths)
        attention_masks = self.gen_attention_masks(sequence_lengths, max_len)
        output = self.bert(input_ids=padded_inputs, attention_mask=attention_masks)
        hidden_state = output.last_hidden_state
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)

        out = self.dropout(pooler)

        out = self.classifier(out)

        loss = self.criterion(out, targets)
        return loss

    def forward_decode(self, input):
        # The input should be padded, so all samples should have the same length
        sequence_lengths = [len(i) for i in input]
        padded_inputs = pad_sequence(
            [torch.LongTensor(i).to(device) for i in input], batch_first=True
        )

        max_len = max(sequence_lengths)
        attention_masks = self.gen_attention_masks(sequence_lengths, max_len)
        output = self.bert(input_ids=padded_inputs, attention_mask=attention_masks)
        hidden_state = output.last_hidden_state
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)

        out = self.dropout(pooler)

        out = self.classifier(out)

        predicted_labels = torch.argmax(out, dim=1)
        return predicted_labels


class SpeechActBERTLSTM(nn.Module):
    N_UNITS_BERT_OUT = 768

    def __init__(
        self,
        vocab_size,
        n_input_layer_units,
        n_hidden_units_utterance_lstm,
        dropout,
        label_size,
        finetune_bert=True,
    ):
        super(SpeechActBERTLSTM, self).__init__()
        self.ntoken = vocab_size
        self.dropout = nn.Dropout(dropout)

        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.utterance_embedding = torch.nn.Linear(
            self.N_UNITS_BERT_OUT, n_input_layer_units
        )

        self.lstm_utterance = LSTM(
            n_input_layer_units, n_hidden_units_utterance_lstm, 1
        )

        self.decoder = nn.Linear(n_hidden_units_utterance_lstm, label_size)

        self.crf = CRF(label_size)

        self.n_hidden_units_utterance_lstm = n_hidden_units_utterance_lstm

        if not finetune_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def gen_attention_masks(self, sequence_lengths, max_len):
        return torch.tensor(
            [int(s) * [1] + (max_len - int(s)) * [0] for s in sequence_lengths]
        ).to(device)

    def forward_nn(self, input):
        sequence_lengths = [len(i) for i in input]
        padded_inputs = pad_sequence(
            [torch.LongTensor(i).to(device) for i in input], batch_first=True
        )

        max_len = max(sequence_lengths)
        attention_masks = self.gen_attention_masks(sequence_lengths, max_len)
        out_bert = self.bert(input_ids=padded_inputs, attention_mask=attention_masks)
        out_bert = out_bert.last_hidden_state
        out_bert = out_bert[:, 0]

        utterance_embedding = self.utterance_embedding(out_bert)
        utterance_embedding = self.dropout(utterance_embedding)

        hidden_utterance_lstm = self.init_hidden(
            1, 1, self.n_hidden_units_utterance_lstm
        )

        utterance_representations = utterance_embedding.unsqueeze(
            1
        )  # add batch size dimension
        output_utterance_level, hidden_utterance_lstm = self.lstm_utterance(
            utterance_representations, hidden_utterance_lstm
        )

        outputs = self.decoder(output_utterance_level.squeeze(1))

        outputs = outputs.unsqueeze(1)
        return outputs

    def forward(self, inputs, targets):
        outputs = self.forward_nn(inputs)

        targets = targets.unsqueeze(1)

        log_likelihood = self.crf.forward(outputs, targets, reduction="token_mean")

        loss = -log_likelihood

        return loss

    def forward_decode(self, input):
        outputs = self.forward_nn(input)

        labels = self.crf.decode(outputs)

        return labels[0]

    def init_hidden(self, n_layers, batch_size, n_hidden_units):
        parameters_input = next(self.parameters())
        return (
            parameters_input.new_zeros(n_layers, batch_size, n_hidden_units),
            parameters_input.new_zeros(n_layers, batch_size, n_hidden_units),
        )
