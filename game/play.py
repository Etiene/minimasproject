# Input reading class
#
# Adapdated from example by Facebook Research
# https://github.com/facebookresearch/EGG/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from data_reader import PrepareDataset
import egg.core as core

def get_params(params):
    default_train_data_path = "data/train.txt"
    default_validation_data_path = "data/validation.txt"
    default_sender_gumbel_softmax_temperature = 1.0
    default_receiver_cell = "rnn" #{rnn, gru, lstm}
    default_sender_cell = "rnn"
    default_sender_hidden_layers = 10
    default_receiver_hidden_layers = 10
    default_embedding_dim = 10
    default_max_len = 1

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data", type=str, default=default_train_data_path, help="Path to the train data (default: {})".format(default_train_data_path)
    )
    parser.add_argument(
        "--validation_data", type=str, default=default_validation_data_path, help="Path to the validation data (default: {})".format(default_validation_data_path)
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=default_sender_gumbel_softmax_temperature,
        help="GS temperature for the sender, only relevant in Gumbel-Softmax (gs) mode (default: {})".format(default_sender_gumbel_softmax_temperature),
    )
    parser.add_argument(
        "--sender_cell",
        type=str,
        default=default_sender_cell,
        help="Type of the cell used for Sender [rnn, gru, lstm] (default: {})".format(default_sender_cell),
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default=default_receiver_cell,
        help="Type of the cell used for Receiver [rnn, gru, lstm] (default: {})".format(default_receiver_cell),
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=default_sender_hidden_layers,
        help="Size of the hidden layer of Sender (default: {})".format(default_sender_hidden_layers),
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=default_receiver_hidden_layers,
        help="Size of the hidden layer of Receiver (default: {})".format(default_receiver_hidden_layers),
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=default_embedding_dim,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: {})".format(default_embedding_dim),
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=default_embedding_dim,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: {})".format(default_embedding_dim),
    )
    parser.add_argument(
        "--default_max_len",
        type=int,
        default=default_sender_hidden_layers,
        help="Max length of message (default: {})".format(default_max_len),
    )
    args = core.init(parser, params)
    return args

def loss(receiver_output, labels):
    acc = (receiver_output.argmax(dim=1) == labels).detach().float()
    loss = F.mse_loss(receiver_output, labels)
    return loss, {"acc": acc}

class LayerWrapper(nn.Module):
    def __init__(self, layer):
        super(LayerWrapper, self).__init__()
        self.layer = layer

    def forward(self, x, _):
        return self.layer(x)

class Game():
    def __init__(self, params):
        self.opts = get_params(params)

    def create_loader(self, path):
        dataset = PrepareDataset(path=path)
        self.n_features = dataset.get_n_features()
        return DataLoader(
            dataset,
            batch_size=self.opts.batch_size,
            shuffle=True,
            num_workers=1,
        )

    def create_sender(self):
        sender_layer = LayerWrapper(nn.Linear(self.n_features, self.opts.sender_hidden))
        agent = core.RnnSenderGS(
            sender_layer,
            vocab_size=self.opts.vocab_size,
            embed_dim=self.opts.sender_embedding,
            hidden_size=self.opts.sender_hidden,
            cell=self.opts.sender_cell,
            max_len=self.opts.max_len,
            temperature=self.opts.temperature,
        )
        return agent

    def create_receiver(self):
        receiver_layer = LayerWrapper(nn.Linear(self.n_features, self.opts.receiver_hidden))
        agent = core.RnnReceiverGS(
            receiver_layer,
            vocab_size=self.opts.vocab_size,
            embed_dim=self.opts.receiver_embedding,
            hidden_size=self.opts.receiver_hidden,
            cell=self.opts.receiver_cell,
        )
        return agent


    def create_trainer(self, game, sender, optimizer):
        callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)]
        print(game)
        print(self.train_loader)
        print(self.test_loader)
        print(callbacks)

        return core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=self.train_loader,
            validation_data=self.test_loader,
            callbacks=callbacks
            + [core.ConsoleLogger(print_train_loss=True, as_json=True)],
        )

    def prepare_data(self):
        print("Preparing dataset...")
        self.train_loader = self.create_loader(self.opts.train_data)
        self.test_loader = self.create_loader(self.opts.validation_data)

    def build_model(self):
        print("Building model...")
        sender = self.create_sender()
        receiver = self.create_receiver()
        game = core.SenderReceiverRnnGS(sender, receiver, loss)
        optimizer = core.build_optimizer(game.parameters())
        self.trainer = self.create_trainer(game, sender, optimizer)

    def train(self):
        print("Training...")
        self.trainer.train(n_epochs=self.opts.n_epochs)

    def play(self):
        self.prepare_data()
        self.build_model()
        self.train()

if __name__ == "__main__":
    import sys
    game = Game(sys.argv[1:])
    game.play()
