# Mini multi-agent system project

Created for a PhD application process

Two neural networks, a sender and a receiver, work together to get the
sum of two numbers. The sender gets two numbers as input and sends a discrete
message to the receiver, who must complete the task and tell us the sum. These
networks are created using [PyTorch][pytorch] and then are wrapped by the
[EGG framework][egg] to handle the messaging part.

Adapted from [basic games example][example] by Facebook Research.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

[egg]:https://github.com/facebookresearch/EGG
[example]:https://github.com/facebookresearch/EGG/tree/3834759306d7155b9f3182e4b0606f61035c7fed/egg/zoo/basic_games
[pytorch]:https://pytorch.org/
