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
import torch.nn.functional as F
from omegaconf import DictConfig

from openspeech.modules import JasperBlock, JasperSubBlock, MaskConv1d


class Jasper(nn.Module):
    r"""
    Jasper (Just Another Speech Recognizer), an ASR model comprised of 54 layers proposed by NVIDIA.
    Jasper achieved sub 3 percent word error rate (WER) on the LibriSpeech dataset.

    Args:
        num_classes (int): number of classification
        version (str): version of jasper. Marked as BxR: B - number of blocks, R - number of sub-blocks

    Inputs: inputs, input_lengths, residual
        - **inputs**: tensor contains input sequence vector
        - **input_lengths**: tensor contains sequence lengths

    Returns:
            (Tensor, Tensor):

            * outputs (torch.FloatTensor): Log probability of model predictions.  ``(batch, seq_length, num_classes)``
            * output_lengths (torch.LongTensor): The length of output tensor ``(batch)``

    Reference:
        Jason Li. et al.: Jasper: An End-to-End Convolutional Neural Acoustic Model
        https://arxiv.org/pdf/1904.03288.pdf
    """

    def __init__(self, configs: DictConfig, input_dim: int, num_classes: int) -> None:
        super(Jasper, self).__init__()
        self.configs = configs
        self.layers = nn.ModuleList()

        in_channels = eval(self.configs.in_channels)
        out_channels = eval(self.configs.out_channels)
        kernel_size = eval(self.configs.kernel_size)
        stride = eval(self.configs.stride)
        dilation = eval(self.configs.dilation)
        dropout_p = eval(self.configs.dropout_p)

        self.layers.append(
            JasperSubBlock(
                in_channels=input_dim,
                out_channels=out_channels[0],
                kernel_size=kernel_size[0],
                stride=stride[0],
                dilation=dilation[0],
                dropout_p=dropout_p[0],
                activation="relu",
                bias=False,
            )
        )
        self.layers.extend(
            [
                JasperBlock(
                    num_sub_blocks=self.configs.num_sub_blocks,
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    kernel_size=kernel_size[i],
                    dilation=dilation[i],
                    dropout_p=dropout_p[i],
                    activation="relu",
                    bias=False,
                )
                for i in range(1, self.configs.num_blocks + 1)
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
                for i in range(self.configs.num_blocks + 1, self.configs.num_blocks + 4)
            ]
        )

        self.residual_connections = self._create_jasper_dense_residual_connections()

    def count_parameters(self) -> int:
        r"""Count parameters of model"""
        return sum([p.numel for p in self.parameters()])

    def update_dropout(self, dropout_p: float) -> None:
        r"""Update dropout probability of model"""
        for name, child in self.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_p

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward propagate a `inputs` for  encoder_only training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor):

            * outputs (torch.FloatTensor): Log probability of model predictions.  ``(batch, seq_length, num_classes)``
            * output_lengths (torch.LongTensor): The length of output tensor ``(batch)``
        """
        residual, prev_outputs, prev_output_lengths = None, list(), list()
        inputs = inputs.transpose(1, 2)

        for i, layer in enumerate(self.layers[:-1]):
            inputs, input_lengths = layer(inputs, input_lengths, residual)
            prev_outputs.append(inputs)
            prev_output_lengths.append(input_lengths)
            residual = self._get_jasper_dencse_residual(prev_outputs, prev_output_lengths, i)

        outputs, output_lengths = self.layers[-1](inputs, input_lengths, residual)

        for i, layer in enumerate(self.postprocess_layers):
            outputs, output_lengths = layer(outputs, output_lengths)

        outputs = F.log_softmax(outputs.transpose(1, 2), dim=-1)

        return outputs, output_lengths

    def _get_jasper_dencse_residual(self, prev_outputs: list, prev_output_lengths: list, index: int):
        residual = None

        for item in zip(prev_outputs, prev_output_lengths, self.residual_connections[index]):
            prev_output, prev_output_length, residual_modules = item
            conv1x1, batch_norm = residual_modules

            if residual is None:
                residual = conv1x1(prev_output, prev_output_length)[0]
            else:
                residual += conv1x1(prev_output, prev_output_length)[0]

            residual = batch_norm(residual)

        return residual

    def _create_jasper_dense_residual_connections(self) -> nn.ModuleList:
        residual_connections = nn.ModuleList()

        for i in range(self.configs.num_blocks):
            residual_modules = nn.ModuleList()
            for j in range(1, i + 2):
                residual_modules.append(
                    nn.ModuleList(
                        [
                            MaskConv1d(self.configs.in_channels[j], self.configs.out_channels[j], kernel_size=1),
                            nn.BatchNorm1d(self.configs.out_channels[i], eps=1e-03, momentum=0.1),
                        ]
                    )
                )
            residual_connections.append(residual_modules)

        return residual_connections
