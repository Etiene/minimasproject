# Input reading class
#
# Adapdated from example by Facebook Research
# https://github.com/facebookresearch/EGG/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from egg.core import PrintValidationEvents
from egg.core.interaction import Interaction


class CustomPrintValidationEvents(PrintValidationEvents):
    def __init__(self, n_epochs):
        super().__init__(n_epochs)

    @staticmethod
    def print_events(logs: Interaction):
        # Print each line from validation set, categorized, so it can be seen
        # side by side, instead of printing all of the inputs, all of the outputs,
        # etc
        for i in range(len(logs.receiver_output)):
            print("Input {}\t-> Message {}\t-> Output {}\t(Label was: {})".format(
                logs.sender_input[i].tolist(),
                logs.message[i].tolist(),
                [round(n, 3) for n in logs.receiver_output[i].tolist()], # rounding output
                logs.labels[i].tolist()))

        # Prints unique symbols found in messages
        unique_symbols = { symbol.item() for message in logs.message for symbol in message}
        print("Number of unique symbols: {} ({})".format(len(unique_symbols), unique_symbols))
