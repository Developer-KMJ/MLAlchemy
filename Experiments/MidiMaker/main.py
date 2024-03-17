from random import random, randrange
from typing import Tuple
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt

from numpy.lib.stride_tricks import sliding_window_view

import midiplayer
from midiplayer import play_song, load_training_data
from model import RnnModel
from tqdm import tqdm

cwd = os.path.dirname(__file__)


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


def get_batch(vectorized_songs, seq_len, batch_size) -> Tuple[np.array, np.array]:
    # length of vectorized song string
    n = vectorized_songs.shape[0] - 1

    # randomly choose the staring indices for the examples in the training batch
    idx = np.random.choice(n - seq_len, batch_size)

    # windows = sliding_window_view(vectorized_songs, window_shape=seq_len)
    # input_batch = windows[idx]
    # output_batch = windows[idx + 1]

    input_batch = [vectorized_songs[i: i + seq_len] for i in idx]
    output_batch = [vectorized_songs[i + 1: i + seq_len + 1] for i in idx]

    x_batch = np.reshape(input_batch, [batch_size, seq_len])
    y_batch = np.reshape(output_batch, [batch_size, seq_len])

    return x_batch, y_batch

def main2():
    # Download the dataset
    songs = load_training_data()

    # Print one of the songs to inspect it in greater detail!
    example_song = songs[0]
    print("\nExample song: ")
    print(example_song)

    # play_song(example_song)

    # Join our list of song strings into a single string containing all songs
    # songs_joined = join_headerless_songs(songs)
    songs_joined = "\n\n".join(songs)
    # Find all unique characters in the joined string
    vocab = sorted(set(songs_joined))
    print("There are", len(vocab), "unique characters in the dataset")

    # char to index and index to char maps
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = {i: ch for i, ch in enumerate(vocab)}

    # convert data from chars to indices
    vectorized_songs = np.array([char2idx[char] for char in songs_joined])

    print(f'{repr(songs_joined[:10])} ---- characters mapped to int ----> {vectorized_songs[:10]}')

    # check that vectorized_songs is a numpy array
    # assert isinstance(vectorized_songs, np.ndarray), "returned result should be a numpy array"

    # batch_test(vectorized_songs, get_batch)

    model = RnnModel(len(vocab), embedding_size=256, rnn_unit_count=1024, batch_size=32)
    print(model)

    x, y = get_batch(vectorized_songs, seq_len=100, batch_size=32)

    t = torch.from_numpy(x).to(model.device)
    predictions = model(t)
    # predictions = torch.softmax(predictions, 1)
    print("Input shape:      ", t.shape, " # (batch_size, sequence_length)")
    print("Prediction shape: ", predictions.shape, "# (batch_size, sequence_length, vocab_size)")

    sampled_indices = torch.multinomial(predictions[0], 1, replacement=True, generator=None)
    sampled_indices = torch.squeeze(sampled_indices).to('cpu').numpy()
    print(sampled_indices)

    print("Input: \n", repr("".join(idx2char[x[0]])))
    print()
    print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))

    # Flatten the values for the loss, so that cross entropy can be used.

    # Reshape so that the sequence is 1 character per row.
    y1 = torch.from_numpy(y).to(model.device).to(dtype=torch.int64).flatten()

    # reshape so that the predictions are one character prediction per row. (batch_size * sequence_length, num_classes)
    predictions1 = predictions.view(-1, len(vocab))
    example_batch_loss = nn.CrossEntropyLoss(reduction="mean")(predictions1, y1)
    print(f"Prediction shape:{predictions.shape}")
    print(f"Scalar loss:{example_batch_loss.to('cpu').detach().numpy().mean()}")


def generate_text(model, char2idx, idx2char, start_string, generation_length=1000):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = torch.Tensor(input_eval).to(dtype=torch.int64).to(device='cuda')
    input_eval = torch.unsqueeze(input_eval, 0)

    text_generated = []

    hidden_state = None
    for i in range(generation_length):
        predictions, hidden_state = model(input_eval, hidden_state)

        predictions = torch.squeeze(predictions, 0)

        predictions = F.softmax(predictions, dim=1)
        input_eval = torch.multinomial(predictions, num_samples=1)
        predicted_id = input_eval.squeeze(0).to(device='cpu').numpy()[0]

        # predicted_id_tensor = torch.tensor([predicted_id])
        # input_eval = torch.unsqueeze(predicted_id_tensor, 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)


def main():
    # HyperParameters
    epochs = 100  # max number of epochs
    num_training_iterations = 2000  # Increase this to train longer
    batch_size = 4  # Experiment between 1 and 64
    learning_rate = 5e-3  # Experiment between 1e-5 and 1e-1
    seq_length = 100  # Experiment between 50 and 500

    vocab_size = 0  # Overwrite later.
    embedding_dim = 256
    rnn_unit_count = 1024  # Experiment between 1 and 2048
    num_layers = 1

    # Checkpoint location:
    load_chk = False
    model_save_dir = 'training_checkpoints/'
    model_file = "rnnmodel.ckpt"

    # Load the data
    songs = load_training_data("irish.abc")
    songs_joined = "\n\n".join(songs)
    vocab = sorted(set(songs_joined))
    data_size = len(songs_joined)
    vocab_size = len(vocab)

    print(f"There are {vocab_size} unique characters in the dataset")

    # char to index and index to char maps
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = {i: ch for i, ch in enumerate(vocab)}

    # convert data from chars to indices
    vectorized_songs = np.array([char2idx[char] for char in songs_joined])

    # model instance
    model = RnnModel(vocab_size, embedding_dim, rnn_unit_count, num_layers)
    model = model.to(model.device)

    # data tensor on device
    vectorized_songs = torch.tensor(vectorized_songs).to(dtype=torch.int64).to(model.device)
    vectorized_songs = torch.unsqueeze(vectorized_songs, dim=1)

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    if load_chk:
        model.load_state_dict(torch.load(os.path.join(model_save_dir, model_file)))
        print("Model loaded successfully.")

    history = []

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(1, epochs + 1):

        data_ptr = np.random.randint(100)
        n = 0
        running_loss = 0
        hidden_state = None

        while True:
            # Grab a batch and propagate it through the network
            x_batch = vectorized_songs[data_ptr: data_ptr + seq_length]
            y_batch = vectorized_songs[data_ptr + 1: data_ptr + seq_length + 1]

            # forward pass
            y_hat, hidden_state = model(x_batch, hidden_state)

            # Get current predictions and reshape (flatten) to work with the crossentropyloss function
            # reshape so that the predictions are one character prediction per row.
            # (batch_size * sequence_length, num_classes)
            # Reshape so that the sequence is 1 character per row.
            # y_hat = y_hat.view(-1, len(vocab))
            # y_batch = torch.from_numpy(y_batch).to(model.device).to(dtype=torch.int64).flatten()
            loss = loss_fn(torch.squeeze(y_hat), torch.squeeze(y_batch))
            running_loss += loss.item()

            # Computer gradients and take optimizer step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data_ptr += seq_length
            n += 1

            if data_ptr + seq_length + 1 > data_size:
                break

            history.append(loss.detach().to(device='cpu').numpy().mean())

        # Update the model with the changed weights!
        torch.save(model.state_dict(), os.path.join(model_save_dir, model_file))
        print(f"epoch:{epoch} iterations:{n}")


        generated_text = generate_text(model, char2idx, idx2char, start_string='X', generation_length=1000)
        midiplayer.save_song_to_midi(generated_text, f"epoch-{epoch}")

    torch.save(model.state_dict(), os.path.join(model_save_dir, model_file))

    plt.figure()
    plt.plot(history, label='train loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.close()

    generated_text = generate_text(model, char2idx, idx2char, start_string='X', generation_length=1000)
    midiplayer.save_song_to_midi(generated_text, "final")


if __name__ == '__main__':
    main()
