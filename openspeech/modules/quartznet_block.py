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

from typing import Tuple

import torch
import torch.nn as nn

from openspeech.modules.pointwise_conv1d import PointwiseConv1d
from openspeech.modules.quartznet_subblock import QuartzNetSubBlock


class QuartzNetBlock(nn.Module):
    r"""
    QuartzNet’s design is based on the Jasper architecture, which is a convolutional model trained with
    Connectionist Temporal Classification (CTC) loss. The main novelty in QuartzNet’s architecture is that QuartzNet
    replaced the 1D convolutions with 1D time-channel separable convolutions, an implementation of depthwise separable
    convolutions.

    Inputs: inputs, input_lengths
        inputs (torch.FloatTensor): tensor contains input sequence vector
        input_lengths (torch.LongTensor): tensor contains sequence lengths

    Returns: output, output_lengths
        (torch.FloatTensor, torch.LongTensor)

        * output (torch.FloatTensor): tensor contains output sequence vector
        * output_lengths (torch.LongTensor): tensor contains output sequence lengths
    """
    supported_activations = {
        "hardtanh": nn.Hardtanh(0, 20, inplace=True),
        "relu": nn.ReLU(inplace=True),
        "elu": nn.ELU(inplace=True),
        "leaky_relu": nn.LeakyReLU(inplace=True),
        "gelu": nn.GELU(),
    }

    def __init__(
        self,
        num_sub_blocks: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
    ) -> None:
        super(QuartzNetBlock, self).__init__()
        padding = self._get_same_padding(kernel_size, stride=1, dilation=1)
        self.layers = nn.ModuleList(
            [
                QuartzNetSubBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=bias,
                )
                for i in range(num_sub_blocks)
            ]
        )
        self.conv1x1 = PointwiseConv1d(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)

    def _get_same_padding(self, kernel_size: int, stride: int, dilation: int):
        if stride > 1 and dilation > 1:
            raise ValueError("Only stride OR dilation may be greater than 1")
        return (kernel_size // 2) * dilation

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward propagate of QuartzNet block.

        Inputs: inputs, input_lengths
            inputs (torch.FloatTensor): tensor contains input sequence vector
            input_lengths (torch.LongTensor): tensor contains sequence lengths

        Returns: output, output_lengths
            (torch.FloatTensor, torch.LongTensor)

            * output (torch.FloatTensor): tensor contains output sequence vector
            * output_lengths (torch.LongTensor): tensor contains output sequence lengths
        """
        residual = self.batch_norm(self.conv1x1(inputs))

        for layer in self.layers[:-1]:
            inputs, input_lengths = layer(inputs, input_lengths)

        outputs, output_lengths = self.layers[-1](inputs, input_lengths, residual)

        return outputs, output_lengths
