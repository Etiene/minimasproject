# Game for PhD application
#
# Two neural networks, a sender and a receiver, work together to get the
# sum of two numbers. The sender gets two numbers and sends a discrete message
# to the receiver, who must complete the task and tell us the sum. These networks
# are created using PyTorch and then are wrapped by the EGG framework which handles
# the message part.
#
# Adapdated from example by Facebook Research
# https://github.com/facebookresearch/EGG/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# external code
import egg.core as core
from egg.core import PrintValidationEvents
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from __future__ import absolute_import
import datetime

# local code
try:
    from data_reader import PrepareDataset
    from printer import CustomPrintValidationEvents
    from params import get_parser_params
except ImportError:
    from .data_reader import PrepareDataset
    from .printer import CustomPrintValidationEvents
    from .params import get_parser_params


class LayerWrapper(nn.Module):
    # Wrapper class necessary for creating the agents in EGG
    def __init__(self, output):
        super(LayerWrapper, self).__init__()
        self.output = output

    def forward(self, x, *args):
        # "*args" is necessary because the EGG sender and the EGG receiver call this
        # function with a different number of arguments. We only use x and this
        # inner architecture is the same for both agents, so we don't need different
        # wrappes for the sender and the receiver. The EGG architecture is different
        # though, but this is set in the create functions of the Game class below.
        return self.output(x)


class Game():
    # Main game
    def __init__(self, args):
        # EGG depends on argparse
        parser, params = get_parser_params(args)
        self.opts = core.init(parser, params)

    def create_data_loader(self, path, shuffle):
        # uses custom data reader to send to torch loader
        dataset = PrepareDataset(path=path)
        self.n_features = dataset.get_n_features()
        return DataLoader(
            dataset,
            batch_size=self.opts.batch_size,
            shuffle=shuffle,
            num_workers=1,
        )

    def create_sender(self):
        # Creating the agent using the layer wrapper and the EGG architecture
        # Using Reinforcing agent.
        arch = nn.Linear(self.n_features, self.opts.sender_hidden)
        layer = LayerWrapper(arch)
        agent = core.RnnSenderReinforce(
            layer,
            vocab_size=self.opts.vocab_size,
            embed_dim=self.opts.sender_embedding,
            hidden_size=self.opts.sender_hidden,
            cell=self.opts.sender_cell,
            max_len=self.opts.max_len,  # 1 is default
        )
        return agent

    def create_receiver(self):
        # Creating the agent using the wrapper and the EGG architecture
        # Using a deterministic agent (output is always the same for a certain
        # message). Output is size 1, a tensor with just the number we want it
        # to guess as we are using a continuous loss function.
        arch = nn.Linear(self.opts.receiver_hidden, 1)
        layer = LayerWrapper(arch)
        agent = core.RnnReceiverDeterministic(
            layer,
            vocab_size=self.opts.vocab_size,
            embed_dim=self.opts.receiver_embedding,
            hidden_size=self.opts.receiver_hidden,
            cell=self.opts.receiver_cell,
        )
        return agent

    def create_trainer(self, game, sender):
        optimizer = core.build_optimizer(self.game.parameters())

        # Test with a scheduler
        if(self.opts.use_scheduler is True):
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.opts.n_epochs
            )

        return core.Trainer(
            game=game,
            optimizer=optimizer,
            optimizer_scheduler=self.opts.use_scheduler if self.opts.use_scheduler else None,
            train_data=self.train_loader,
            validation_data=self.test_loader,
            callbacks=[
                core.ConsoleLogger(print_train_loss=True, as_json=True),
                # Custom printer made to see information of each line side by side
                CustomPrintValidationEvents(n_epochs=self.opts.n_epochs)
            ],
        )

    def prepare_data(self):
        print("Preparing dataset...")
        self.train_loader = self.create_data_loader(self.opts.train_data, True)
        print("Samples in training set: {}".format(
            len(self.train_loader.dataset.frame)))
        self.test_loader = self.create_data_loader(
            self.opts.validation_data, False)
        print("Samples in test set: {}".format(
            len(self.test_loader.dataset.frame)))

    # This loss function is sent to the EGG game constructor and must have this
    # exact number of parameters. That means it cant receive 'self', hence its
    # defined as static
    @staticmethod
    def _loss(sender_input, _message, _receiver_input, receiver_output, labels, _aux_input):
        # Using torch.isclose as a rough measure of accuracy since rounding
        # output in a continuous loss function will prevent model from training.
        # Absolute tolerance is set to 0.5 so that if output is 6.01, and label is
        # 6, that is considered good enough as the nearest integer is the label.
        # https://pytorch.org/docs/stable/generated/torch.isclose.html#torch.isclose
        acc = receiver_output.isclose(labels, atol=0.5).detach().float()
        # Mean Squared Error loss, for continuous output
        loss = F.mse_loss(receiver_output, labels, reduction="none")
        # Averaging for batches biggers than 1
        loss = loss.mean()
        return loss, {"acc": acc}

    def build_model(self):
        print("\n----------")
        print("Building model...")

        sender = self.create_sender()
        receiver = self.create_receiver()
        self.game = core.SenderReceiverRnnReinforce(
            sender,
            receiver,
            self._loss,
            sender_entropy_coeff=self.opts.sender_entropy,
        )
        self.trainer = self.create_trainer(self.game, sender)

        print(self.game)
        print("Number of parameters {}".format(
            nn.utils.parameters_to_vector(self.game.parameters()).numel()))

    def train(self):
        print("\n----------")
        a = datetime.datetime.now()
        print("Start of training... {}".format(a))
        self.trainer.train(n_epochs=self.opts.n_epochs)
        b = datetime.datetime.now()
        print("End of training... {}".format(b))
        print("Duration: {} seconds".format(round((b-a).total_seconds(), 2)))

    def play(self):
        self.prepare_data()
        self.build_model()
        self.train()


if __name__ == "__main__":
    import sys
    game = Game(sys.argv[1:])
    game.play()
