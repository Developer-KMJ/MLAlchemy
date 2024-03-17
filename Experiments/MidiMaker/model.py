import torch
import torch.nn as nn


class RnnModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_size, rnn_unit_count, num_layers=1):
        super(RnnModel, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.rnn_unit_count = rnn_unit_count
        self.num_layers = num_layers

        # Layer 1: Embedding layer to transform indices into dense vectors of a fixed embedding size
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)

        # Layer 2: LSTM with `rnn_units` number of units.
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=rnn_unit_count, num_layers=self.num_layers)

        # Layer 3: Dense (fully-connected) layer that transforms the LSTM output into the vocabulary size.
        self.decoder = nn.Linear(rnn_unit_count, vocabulary_size)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_seq, hidden_state):
        embedding = self.embedding(input_seq)
        lstm_output, hidden_state = self.lstm(embedding, hidden_state)
        model_output = self.decoder(lstm_output)

        return model_output, (hidden_state[0].detach(), hidden_state[1].detach())


#
# class RnnModel(nn.Module):
#     def __init__(self, vocabulary_size, embedding_size, rnn_units, batch_size):
#         super(RnnModel, self).__init__()
#         self.vocabulary_size = vocabulary_size
#         self.embedding_size = embedding_size
#         self.rnn_units = rnn_units
#         self.batch_size = batch_size
#
#         # Layer 1: Embedding layer to transform indices into dense vectors of a fixed embedding size
#         self.embedding_layer = nn.Embedding(vocabulary_size, embedding_size)
#
#         # Layer 2: LSTM with `rnn_units` number of units.
#         # self.lstm_layer = nn.LSTM(embedding_size, rnn_units, 1, batch_first=True)
#         self.lstm_layer = nn.LSTM(embedding_size, rnn_units)
#         self.sigmoid_layer = nn.Sigmoid()
#
#         # Layer 3: Dense (fully-connected) layer that transforms the LSTM output into the vocabulary size.
#         self.linear_layer = nn.Linear(rnn_units, vocabulary_size)
#
#       #  self.softmax_layer = nn.Softmax(1)
#
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.to(self.device)
#
#     def forward(self, data_input):
#         lstm_input = self.embedding_layer(data_input)
#         lstm_output, _ = self.lstm_layer(lstm_input)
#       #  model_outputs = self.linear_layer(lstm_output)
#         sigmoid_output = self.sigmoid_layer(lstm_output)
#         model_outputs = self.linear_layer(sigmoid_output)
#
#         return model_outputs
#       #  softmax_outputs = self.softmax_layer(model_outputs)
#       #  return softmax_outputs
