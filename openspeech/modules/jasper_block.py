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

import torch.nn as nn
from torch import Tensor

from openspeech.modules.jasper_subblock import JasperSubBlock


class JasperBlock(nn.Module):
    r"""
    Jasper Block: The Jasper Block consists of R Jasper sub-block.

    Args:
        num_sub_blocks (int): number of sub block
        in_channels (int): number of channels in the input feature
        out_channels (int): number of channels produced by the convolution
        kernel_size (int): size of the convolving kernel
        stride (int): stride of the convolution. (default: 1)
        dilation (int): spacing between kernel elements. (default: 1)
        bias (bool): if True, adds a learnable bias to the output. (default: True)
        dropout_p (float): probability of dropout
        activation (str): activation function

    Inputs: inputs, input_lengths, residual
        - **inputs**: tensor contains input sequence vector
        - **input_lengths**: tensor contains sequence lengths
        - **residual**: tensor contains residual vector

    Returns: output, output_lengths
        (torch.FloatTensor, torch.LongTensor)

        * output (torch.FloatTensor): tensor contains output sequence vector
        * output_lengths (torch.LongTensor): tensor contains output sequence lengths
    """

    def __init__(
        self,
        num_sub_blocks: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        dropout_p: float = 0.2,
        activation: str = "relu",
    ) -> None:
        super(JasperBlock, self).__init__()
        padding = self._get_same_padding(kernel_size, stride, dilation)
        self.layers = nn.ModuleList(
            [
                JasperSubBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    dropout_p=dropout_p,
                    activation=activation,
                )
                for i in range(num_sub_blocks)
            ]
        )

    def _get_same_padding(self, kernel_size: int, stride: int, dilation: int):
        if stride > 1 and dilation > 1:
            raise ValueError("Only stride OR dilation may be greater than 1")
        return (kernel_size // 2) * dilation

    def forward(self, inputs: Tensor, input_lengths: Tensor, residual: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Forward propagate of jasper block.

        Inputs: inputs, input_lengths, residual
            - **inputs**: tensor contains input sequence vector
            - **input_lengths**: tensor contains sequence lengths
            - **residual**: tensor contains residual vector

        Returns: output, output_lengths
            (torch.FloatTensor, torch.LongTensor)

            * output (torch.FloatTensor): tensor contains output sequence vector
            * output_lengths (torch.LongTensor): tensor contains output sequence lengths
        """
        for layer in self.layers[:-1]:
            inputs, input_lengths = layer(inputs, input_lengths)

        outputs, output_lengths = self.layers[-1](inputs, input_lengths, residual)

        return outputs, output_lengths
