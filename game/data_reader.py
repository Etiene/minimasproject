# Input reading class
#
# Adapdated from example by Facebook Research
# https://github.com/facebookresearch/EGG/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils.data import Dataset

# This class opents a text file containing three whole numbers per line and
# converts it to the the necessary format for the simulation with EGG.
# The two first numbers are the ones that need to be summed, and the last
# number is the label sum.
class PrepareDataset(Dataset):
    def __init__(self, path):
        frame = open(path, "r")
        self.frame = []
        for row in frame:
            row = row.split(" ")
            numbers = list(map(int, row))
            if(len(numbers) != 3):
                raise ValueError("Error reading data, unexpected length {} of numbers on line {}. Must be 3.".format(len(numbers), numbers))
            input = numbers[:2]
            label = numbers[2]
            self.frame.append((torch.tensor(input), torch.tensor(label)))


    def get_n_features(self):
        # TODO: review this function
        return self.frame[0][0].size(0)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        return self.frame[idx]
