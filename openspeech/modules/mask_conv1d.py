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

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class MaskConv1d(nn.Conv1d):
    r"""
    1D convolution with masking

    Args:
        in_channels (int): Number of channels in the input vector
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int): Stride of the convolution. Default: 1
        padding (int):  Zero-padding added to both sides of the input. Default: 0
        dilation (int): Spacing between kernel elements. Default: 1
        groups (int): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool): If True, adds a learnable bias to the output. Default: True

    Inputs: inputs, seq_lengths
        - **inputs** (torch.FloatTensor): The input of size (batch, dimension, time)
        - **seq_lengths** (torch.IntTensor): The actual length of each sequence in the batch

    Returns: output, seq_lengths
        - **output**: Masked output from the conv1d
        - **seq_lengths**: Sequence length of output from the conv1d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super(MaskConv1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def _get_sequence_lengths(self, seq_lengths):
        return (seq_lengths + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[
            0
        ] + 1

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        inputs: (batch, dimension, time)
        input_lengths: (batch)
        """
        max_length = inputs.size(2)

        indices = torch.arange(max_length).to(input_lengths.dtype).to(input_lengths.device)
        indices = indices.expand(len(input_lengths), max_length)

        mask = indices >= input_lengths.unsqueeze(1)
        inputs = inputs.masked_fill(mask.unsqueeze(1).to(device=inputs.device), 0)

        output_lengths = self._get_sequence_lengths(input_lengths)
        output = super(MaskConv1d, self).forward(inputs)

        del mask, indices

        return output, output_lengths
