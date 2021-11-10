# Function for getting params
#
# Adapdated from example by Facebook Research
# https://github.com/facebookresearch/EGG/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse


def get_parser_params(params):
    default_train_data_path = "data/train.txt"
    default_validation_data_path = "data/validation.txt"
    default_receiver_cell = "rnn"  # {rnn, gru, lstm}
    default_sender_cell = "rnn"
    default_sender_hidden_layers = 10
    default_receiver_hidden_layers = 10
    default_embedding_dim = 10
    default_sender_entropy = 0.8
    default_use_scheduler = False

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data", type=str, default=default_train_data_path, help="Path to the train data (default: {})".format(default_train_data_path)
    )
    parser.add_argument(
        "--validation_data", type=str, default=default_validation_data_path, help="Path to the validation data (default: {})".format(default_validation_data_path)
    )
    parser.add_argument(
        "--sender_entropy",
        type=float,
        default=default_sender_cell,
        help="The entropy coefficient for the sender reinforcement (default: {})".format(
            default_sender_entropy),
    )
    parser.add_argument(
        "--sender_cell",
        type=str,
        default=default_sender_cell,
        help="Type of the cell used for Sender [rnn, gru, lstm] (default: {})".format(
            default_sender_cell),
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default=default_receiver_cell,
        help="Type of the cell used for Receiver [rnn, gru, lstm] (default: {})".format(
            default_receiver_cell),
    )
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=default_sender_hidden_layers,
        help="Size of the hidden layer of Sender (default: {})".format(
            default_sender_hidden_layers),
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=default_receiver_hidden_layers,
        help="Size of the hidden layer of Receiver (default: {})".format(
            default_receiver_hidden_layers),
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=default_embedding_dim,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: {})".format(
            default_embedding_dim),
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=default_embedding_dim,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: {})".format(
            default_embedding_dim),
    )
    parser.add_argument(
        "--use_scheduler", type=bool, default=default_use_scheduler, help="Use learning rate scheduler (default: {})".format(default_use_scheduler)
    )

    return (parser, params)
