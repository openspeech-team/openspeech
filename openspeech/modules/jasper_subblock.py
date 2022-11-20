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

from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor

from openspeech.modules.mask_conv1d import MaskConv1d


class JasperSubBlock(nn.Module):
    r"""
    Jasper sub-block applies the following operations: a 1D-convolution, batch norm, ReLU, and dropout.

    Args:
        in_channels (int): number of channels in the input feature
        out_channels (int): number of channels produced by the convolution
        kernel_size (int): size of the convolving kernel
        stride (int): stride of the convolution. (default: 1)
        dilation (int): spacing between kernel elements. (default: 1)
        padding (int): zero-padding added to both sides of the input. (default: 0)
        bias (bool): if True, adds a learnable bias to the output. (default: False)
        dropout_p (float): probability of dropout
        activation (str): activation function

    Inputs: inputs, input_lengths, residual
        - **inputs**: tensor contains input sequence vector
        - **input_lengths**: tensor contains sequence lengths
        - **residual**: tensor contains residual vector

    Returns: output, output_lengths
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
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        bias: bool = False,
        dropout_p: float = 0.2,
        activation: str = "relu",
    ) -> None:
        super(JasperSubBlock, self).__init__()

        self.conv = MaskConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        self.batch_norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        self.activation = self.supported_activations[activation]
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
        residual: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Forward propagate of conformer's subblock.

        Inputs: inputs, input_lengths, residual
            - **inputs**: tensor contains input sequence vector
            - **input_lengths**: tensor contains sequence lengths
            - **residual**: tensor contains residual vector

        Returns: output, output_lengths
            * output (torch.FloatTensor): tensor contains output sequence vector
            * output_lengths (torch.LongTensor): tensor contains output sequence lengths
        """
        outputs, output_lengths = self.conv(inputs, input_lengths)
        outputs = self.batch_norm(outputs)

        if residual is not None:
            outputs += residual

        outputs = self.dropout(self.activation(outputs))

        return outputs, output_lengths
