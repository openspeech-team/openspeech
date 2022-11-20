# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch.nn as nn
from torch import Tensor

from openspeech.modules import Conv2dSubsampling, DeepSpeech2Extractor, Swish, VGGExtractor


class OpenspeechEncoder(nn.Module):
    r"""
    Base Interface of Openspeech Encoder.

    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
    """
    supported_activations = {
        "hardtanh": nn.Hardtanh(0, 20, inplace=True),
        "relu": nn.ReLU(inplace=True),
        "elu": nn.ELU(inplace=True),
        "leaky_relu": nn.LeakyReLU(inplace=True),
        "gelu": nn.GELU(),
        "swish": Swish(),
    }
    supported_extractors = {
        "ds2": DeepSpeech2Extractor,
        "vgg": VGGExtractor,
        "conv2d_subsample": Conv2dSubsampling,
    }

    def __init__(self):
        super(OpenspeechEncoder, self).__init__()

    def count_parameters(self) -> int:
        r"""Count parameters of encoders"""
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        r"""Update dropout probability of encoders"""
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        r"""
        Forward propagate for encoders training.

        Inputs:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        """
        raise NotImplementedError
