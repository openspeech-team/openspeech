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

from openspeech.encoders.openspeech_encoder import OpenspeechEncoder
from openspeech.modules import (
    DepthwiseConv2dSubsampling,
    ResidualConnectionModule,
    SqueezeformerBlock,
    TimeReductionLayer,
    recover_resolution,
)


class SqueezeformerEncoder(OpenspeechEncoder):
    """
    Squeezeformer encoder first processes the input with a convolution subsampling layer and then
    with a number of squeezeformer blocks.

    Args:
        num_classes (int): Number of classification
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of squeezeformer encoder
        num_layers (int, optional): Number of squeezeformer blocks
        reduce_layer_index (int, optional): The layer index to reduce sequence length
        recover_layer_index (int, optional): The layer index to recover sequence length
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of squeezeformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of squeezeformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not

    Inputs: inputs, input_lengths
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths

    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by squeezeformer encoder.
        - **output_lengths** (batch): list of sequence output lengths

    Reference:
        Squeezeformer: An Efficient Transformer for Automatic Speech Recognition
        https://arxiv.org/abs/2206.00888
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int = 80,
        encoder_dim: int = 512,
        num_layers: int = 16,
        reduce_layer_index: int = 7,
        recover_layer_index: int = 15,
        num_attention_heads: int = 8,
        feed_forward_expansion_factor: int = 4,
        conv_expansion_factor: int = 2,
        input_dropout_p: float = 0.1,
        feed_forward_dropout_p: float = 0.1,
        attention_dropout_p: float = 0.1,
        conv_dropout_p: float = 0.1,
        conv_kernel_size: int = 31,
        half_step_residual: bool = False,
        joint_ctc_attention: bool = True,
    ):
        super(SqueezeformerEncoder, self).__init__()
        self.num_layers = num_layers
        self.reduce_layer_index = reduce_layer_index
        self.recover_layer_index = recover_layer_index
        self.joint_ctc_attention = joint_ctc_attention
        self.conv_subsample = DepthwiseConv2dSubsampling(input_dim, in_channels=1, out_channels=encoder_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim),
            nn.Dropout(p=input_dropout_p),
        )
        self.time_reduction_layer = TimeReductionLayer()
        self.time_reduction_proj = nn.Linear((encoder_dim - 1) // 2, encoder_dim)
        self.time_recover_layer = nn.Linear(encoder_dim, encoder_dim)
        self.recover_tensor = None

        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            if idx < reduce_layer_index:
                self.layers.append(
                    SqueezeformerBlock(
                        encoder_dim=encoder_dim,
                        num_attention_heads=num_attention_heads,
                        feed_forward_expansion_factor=feed_forward_expansion_factor,
                        conv_expansion_factor=conv_expansion_factor,
                        feed_forward_dropout_p=feed_forward_dropout_p,
                        attention_dropout_p=attention_dropout_p,
                        conv_dropout_p=conv_dropout_p,
                        conv_kernel_size=conv_kernel_size,
                        half_step_residual=half_step_residual,
                    )
                )
            elif reduce_layer_index <= idx < recover_layer_index:
                self.layers.append(
                    ResidualConnectionModule(
                        module=SqueezeformerBlock(
                            encoder_dim=encoder_dim,
                            num_attention_heads=num_attention_heads,
                            feed_forward_expansion_factor=feed_forward_expansion_factor,
                            conv_expansion_factor=conv_expansion_factor,
                            feed_forward_dropout_p=feed_forward_dropout_p,
                            attention_dropout_p=attention_dropout_p,
                            conv_dropout_p=conv_dropout_p,
                            conv_kernel_size=conv_kernel_size,
                            half_step_residual=half_step_residual,
                        )
                    )
                )
            else:
                self.layers.append(
                    SqueezeformerBlock(
                        encoder_dim=encoder_dim,
                        num_attention_heads=num_attention_heads,
                        feed_forward_expansion_factor=feed_forward_expansion_factor,
                        conv_expansion_factor=conv_expansion_factor,
                        feed_forward_dropout_p=feed_forward_dropout_p,
                        attention_dropout_p=attention_dropout_p,
                        conv_dropout_p=conv_dropout_p,
                        conv_kernel_size=conv_kernel_size,
                        half_step_residual=half_step_residual,
                    )
                )

        if self.joint_ctc_attention:
            self.fc = nn.Sequential(
                nn.Dropout(feed_forward_dropout_p),
                nn.Linear(encoder_dim, num_classes, bias=False),
            )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward propagate a `inputs` for  encoder training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            (Tensor, Tensor)
            * outputs (torch.FloatTensor): A output sequence of encoder. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * output_lengths (torch.LongTensor): The length of output tensor. ``(batch)``
        """
        encoder_logits = None

        outputs, output_lengths = self.conv_subsample(inputs, input_lengths)
        outputs = self.input_proj(outputs)

        for idx, layer in enumerate(self.layers):
            if idx == self.reduce_layer_index:
                self.recover_tensor = outputs
                outputs, output_lengths = self.time_reduction_layer(outputs, output_lengths)
                outputs = self.time_reduction_proj(outputs)

            if idx == self.recover_layer_index:
                outputs = recover_resolution(outputs)
                length = outputs.size(1)
                outputs = self.time_recover_layer(outputs)
                outputs += self.recover_tensor[:, :length, :]
                output_lengths *= 2

            outputs = layer(outputs)

        if self.joint_ctc_attention:
            encoder_logits = self.fc(outputs).log_softmax(dim=2)

        return outputs, encoder_logits, output_lengths
