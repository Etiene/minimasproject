# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# usage: python generator.py number_of_samples max_int > file.txt
import random
import sys


def create_numbers(max_limit, n_samples=None, format_for_training=True, echo=False, start_from=0):
    random.seed(a=None, version=2)
    all = {(-1, -1): True}
    lines = []
    inputs = []
    outputs = []
    # checking max number of samples that can be created
    n_integers = max_limit + 1
    max_samples = n_integers * n_integers - sum(range(n_integers))
    if n_samples is None or n_samples > max_samples:
        n_samples = max_samples
    print("Generating {} samples...".format(n_samples))

    for i in range(n_samples):
        output, input_1 = -1, -1
        # avoiding duplicates
        # could halt forever
        while(all.get((output, input_1), None) != None):
            output = random.randint(start_from, max_limit)
            input_1 = random.randint(0, output)
        all[(output, input_1)] = True
        input_2 = output - input_1
        if format_for_training:
            lines.append("{} {} {}".format(input_1, input_2, output))
        else:
            inputs.append([input_1, input_2])
            outputs.append([output])
        if echo:
            print(input_1, input_2, output)

    if format_for_training:
        return lines

    return inputs, outputs


if __name__ == "__main__":
    n_samples = int(sys.argv[1])
    max_limit = int(sys.argv[2]) or 1000
    create_numbers(max_limit, n_samples, True, True)
