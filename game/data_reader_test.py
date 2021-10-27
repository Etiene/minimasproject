# Test for input reading class
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from data_reader import PrepareDataset

dataset = PrepareDataset("./data/train.txt")
print(dataset.frame)
