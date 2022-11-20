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

from openspeech.modules.contextnet_module import ContextNetConvModule, ContextNetSEModule
from openspeech.modules.swish import Swish


class ContextNetBlock(nn.Module):
    r"""
    Convolution block contains a number of convolutions, each followed by batch normalization and activation.
    Squeeze-and-excitation (SE) block operates on the output of the last convolution layer.
    Skip connection with projection is applied on the output of the squeeze-and-excitation block.

    Args:
        in_channels (int): Input channel in convolutional layer
        out_channels (int): Output channel in convolutional layer
        num_layers (int, optional): The number of convolutional layers (default : 5)
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        stride(int, optional): Value of stride (default : 1)
        padding (int, optional): Value of padding (default: 0)
        residual (bool, optional): Flag indication residual or not (default : True)

    Inputs: inputs, input_lengths
        - **inputs**: Input of convolution block `FloatTensor` of size ``(batch, dimension, seq_length)``
        - **input_lengths**: The length of input tensor. ``(batch)``

    Returns: output, output_lengths
        - **output**: Output of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
        - **output_lengths**: The length of output tensor. ``(batch)``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 5,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 0,
        residual: bool = True,
    ) -> None:
        super(ContextNetBlock, self).__init__()
        self.num_layers = num_layers
        self.swish = Swish()
        self.se_layer = ContextNetSEModule(out_channels)
        self.residual = None

        if residual:
            self.residual = ContextNetConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                activation=False,
            )

        if self.num_layers == 1:
            self.conv_layers = ContextNetConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

        else:
            stride_list = [1 for _ in range(num_layers - 1)] + [stride]
            in_channel_list = [in_channels] + [out_channels for _ in range(num_layers - 1)]

            self.conv_layers = nn.ModuleList(list())
            for in_channels, stride in zip(in_channel_list, stride_list):
                self.conv_layers.append(
                    ContextNetConvModule(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                    )
                )

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Forward propagate a `inputs` for convolution block.

        Args:
            **inputs** (torch.FloatTensor): Input of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            **output** (torch.FloatTensor): Output of convolution block `FloatTensor` of size
                ``(batch, dimension, seq_length)``
            **output_lengths** (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        output = inputs
        output_lengths = input_lengths

        if self.num_layers == 1:
            output, output_lengths = self.conv_layers(output, output_lengths)
        else:
            for conv_layer in self.conv_layers:
                output, output_lengths = conv_layer(output, output_lengths)

        output = self.se_layer(output, output_lengths)

        if self.residual is not None:
            residual, _ = self.residual(inputs, input_lengths)
            output += residual

        return self.swish(output), output_lengths

    @staticmethod
    def make_conv_blocks(
        input_dim: int = 80,
        num_layers: int = 5,
        kernel_size: int = 5,
        num_channels: int = 256,
        output_dim: int = 640,
    ) -> nn.ModuleList:
        r"""
        Create 23 convolution blocks.

        Args:
            input_dim (int, optional): Dimension of input vector (default : 80)
            num_layers (int, optional): The number of convolutional layers (default : 5)
            kernel_size (int, optional): Value of convolution kernel size (default : 5)
            num_channels (int, optional): The number of channels in the convolution filter (default: 256)
            output_dim (int, optional): Dimension of encoder output vector (default: 640)

        Returns:
            **conv_blocks** (nn.ModuleList): ModuleList with 23 convolution blocks
        """
        conv_blocks = nn.ModuleList()

        # C0 : 1 conv layer, init_dim output channels, stride 1, no residual
        conv_blocks.append(ContextNetBlock(input_dim, num_channels, 1, kernel_size, 1, 0, False))

        # C1-2 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(1, 2 + 1):
            conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, True))

        # C3 : 5 conv layer, init_dim output channels, stride 2
        conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 2, 0, True))

        # C4-6 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(4, 6 + 1):
            conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, True))

        # C7 : 5 conv layers, init_dim output channels, stride 2
        conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 2, 0, True))

        # C8-10 : 5 conv layers, init_dim output channels, stride 1
        for _ in range(8, 10 + 1):
            conv_blocks.append(ContextNetBlock(num_channels, num_channels, num_layers, kernel_size, 1, 0, True))

        # C11-13 : 5 conv layers, middle_dim output channels, stride 1
        conv_blocks.append(ContextNetBlock(num_channels, num_channels << 1, num_layers, kernel_size, 1, 0, True))
        for _ in range(12, 13 + 1):
            conv_blocks.append(
                ContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 1, 0, True)
            )

        # C14 : 5 conv layers, middle_dim output channels, stride 2
        conv_blocks.append(ContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 2, 0, True))

        # C15-21 : 5 conv layers, middle_dim output channels, stride 1
        for i in range(15, 21 + 1):
            conv_blocks.append(
                ContextNetBlock(num_channels << 1, num_channels << 1, num_layers, kernel_size, 1, 0, True)
            )

        # C22 : 1 conv layer, final_dim output channels, stride 1, no residual
        conv_blocks.append(ContextNetBlock(num_channels << 1, output_dim, 1, kernel_size, 1, 0, False))

        return conv_blocks
