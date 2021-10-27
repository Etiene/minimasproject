# Test for input reading class
#
# Adapdated from example by Facebook Research
# https://github.com/facebookresearch/EGG/
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import data_reader as d

dataset = d.PrepareDataset("./data/train.txt")
print(dataset.frame)
