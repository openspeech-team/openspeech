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
from omegaconf import DictConfig

from openspeech.modules import JasperSubBlock, QuartzNetBlock


class QuartzNet(nn.Module):
    r"""
    QuartzNet is fully convolutional automatic speech recognition model.  The model is composed of multiple
    blocks with residual connections between them. Each block consists of one or more modules with
    1D time-channel separable convolutional layers, batch normalization, and ReLU layers.
    It is trained with CTC loss.

    Args:
        configs (DictConfig): hydra configuration set.
        input_dim (int): dimension of input.
        num_classes (int): number of classification.

    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        (Tensor, Tensor):

        * outputs (torch.FloatTensor): Log probability of model predictions.  ``(batch, seq_length, num_classes)``
        * output_lengths (torch.LongTensor): The length of output tensor ``(batch)``

    Reference:
        Samuel Kriman et al.: QUARTZNET: DEEP AUTOMATIC SPEECH RECOGNITION WITH 1D TIME-CHANNEL SEPARABLE CONVOLUTIONS.
        https://arxiv.org/abs/1910.10261.pdf
    """

    def __init__(self, configs: DictConfig, input_dim: int, num_classes: int) -> None:
        super(QuartzNet, self).__init__()
        self.configs = configs

        in_channels = eval(self.configs.model.in_channels)
        out_channels = eval(self.configs.model.out_channels)
        kernel_size = eval(self.configs.model.kernel_size)
        dilation = eval(self.configs.model.dilation)
        dropout_p = eval(self.configs.model.dropout_p)

        self.preprocess_layer = JasperSubBlock(
            in_channels=input_dim,
            out_channels=out_channels[0],
            kernel_size=kernel_size[0],
            dilation=dilation[0],
            dropout_p=dropout_p[0],
            activation="relu",
            bias=False,
        )
        self.layers = nn.ModuleList(
            [
                QuartzNetBlock(
                    num_sub_blocks=self.configs.model.num_sub_blocks,
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=kernel_size[i],
                    bias=False,
                )
                for i in range(1, self.configs.model.num_blocks + 1)
            ]
        )
        self.postprocess_layers = nn.ModuleList(
            [
                JasperSubBlock(
                    in_channels=in_channels[i],
                    out_channels=num_classes if out_channels[i] is None else out_channels[i],
                    kernel_size=kernel_size[i],
                    dilation=dilation[i],
                    dropout_p=dropout_p[i],
                    activation="relu",
                    bias=True if i == 2 else False,
                )
                for i in range(self.configs.model.num_blocks + 1, self.configs.model.num_blocks + 4)
            ]
        )

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward propagate a `inputs` for  encoder_only training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor):

            * outputs (torch.FloatTensor): Log probability of model predictions.  ``(batch, seq_length, num_classes)``
            * output_lengths (torch.LongTensor): The length of output tensor ``(batch)``
        """
        inputs = inputs.transpose(1, 2)

        outputs, output_lengths = self.preprocess_layer(inputs, input_lengths)

        for layer in self.layers:
            outputs, output_lengths = layer(outputs, output_lengths)

        for layer in self.postprocess_layers:
            outputs, output_lengths = layer(outputs, output_lengths)

        return outputs.transpose(1, 2), output_lengths
