# Test for input reading class
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import data_reader as d

dataset = d.PrepareDataset("./data/train.txt")
print(dataset.frame)
