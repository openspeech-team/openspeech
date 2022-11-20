# MIT License
#
# Copyright (c) 2021 Soohwan Kim
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

from typing import Optional, Tuple

import torch
import torch.nn as nn

from openspeech.modules.conv_group_shuffle import ConvGroupShuffle
from openspeech.modules.depthwise_conv1d import DepthwiseConv1d
from openspeech.modules.time_channel_separable_conv1d import TimeChannelSeparableConv1d


class QuartzNetSubBlock(nn.Module):
    r"""
    QuartzNet sub-block applies the following operations: a 1D-convolution, batch norm, ReLU, and dropout.

    Args:
        in_channels (int): number of channels in the input feature
        out_channels (int): number of channels produced by the convolution
        kernel_size (int): size of the convolving kernel
        padding (int): zero-padding added to both sides of the input. (default: 0)
        bias (bool): if True, adds a learnable bias to the output. (default: False)

    Inputs: inputs, input_lengths, residual
        - **inputs**: tensor contains input sequence vector
        - **input_lengths**: tensor contains sequence lengths
        - **residual**: tensor contains residual vector

    Returns: output, output_lengths
        * output (torch.FloatTensor): tensor contains output sequence vector
        * output_lengths (torch.LongTensor): tensor contains output sequence lengths
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = False,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super(QuartzNetSubBlock, self).__init__()
        self.depthwise_conf1d = DepthwiseConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.tcs_conv = TimeChannelSeparableConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            groups=groups,
            bias=bias,
        )
        self.group_shuffle = ConvGroupShuffle(groups, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        self.relu = nn.ReLU()

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs, output_lengths = self.depthwise_conf1d(inputs, input_lengths)
        outputs, output_lengths = self.tcs_conv(outputs, output_lengths)
        outputs = self.group_shuffle(outputs)
        outputs = self.batch_norm(outputs)

        if residual is not None:
            outputs += residual

        outputs = self.relu(outputs)

        return outputs, output_lengths
