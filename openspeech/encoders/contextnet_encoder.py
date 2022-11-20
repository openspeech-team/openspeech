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

from openspeech.encoders.openspeech_encoder import OpenspeechEncoder
from openspeech.modules.contextnet_block import ContextNetBlock


class ContextNetEncoder(OpenspeechEncoder):
    r"""
    ContextNetEncoder goes through 23 convolution blocks to convert to higher feature values.

    Args:
        num_classes (int): Number of classification
        model_size (str, optional): Size of the model['small', 'medium', 'large'] (default : 'medium')
        input_dim (int, optional): Dimension of input vector (default : 80)
        num_layers (int, optional): The number of convolutional layers (default : 5)
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        num_channels (int, optional): The number of channels in the convolution filter (default: 256)
        output_dim (int, optional): Dimension of encoder output vector (default: 640)
        joint_ctc_attention (bool, optional): flag indication joint ctc attention or not

    Inputs: inputs, input_lengths
        - **inputs**: Parsed audio of batch size number `FloatTensor` of size ``(batch, seq_length, dimension)``
        - **input_lengths**: Tensor representing the sequence length of the input ``(batch)``

    Returns: output, output_lengths
        - **output**: Tensor of encoder output `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        - **encoder_logits**: Log probability of encoders outputs will be passed to CTC Loss.
                If joint_ctc_attention is False, return None.
        - **output_lengths**: Tensor representing the length of the encoder output ``(batch)``
    """
    supported_models = {
        "small": 0.5,
        "medium": 1,
        "large": 2,
    }

    def __init__(
        self,
        num_classes: int,
        model_size: str = "medium",
        input_dim: int = 80,
        num_layers: int = 5,
        kernel_size: int = 5,
        num_channels: int = 256,
        output_dim: int = 640,
        joint_ctc_attention: bool = False,
    ) -> None:
        super(ContextNetEncoder, self).__init__()
        assert model_size in ("small", "medium", "large"), f"{model_size} is not supported."

        alpha = self.supported_models[model_size]

        num_channels = int(num_channels * alpha)
        output_dim = int(output_dim * alpha)

        self.joint_ctc_attention = joint_ctc_attention
        self.blocks = ContextNetBlock.make_conv_blocks(input_dim, num_layers, kernel_size, num_channels, output_dim)
        if self.joint_ctc_attention:
            self.fc = nn.Linear(output_dim, num_classes, bias=False)

    def forward(
        self,
        inputs: Tensor,
        input_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Forward propagate a `inputs` for audio encoder.

        Args:
            **inputs** (torch.FloatTensor): Parsed audio of batch size number `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            **input_lengths** (torch.LongTensor): Tensor representing the sequence length of the input
                `LongTensor` of size ``(batch)``

        Returns:
            **output** (torch.FloatTensor): Tensor of encoder output `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            **encoder_logits** (torch.FloatTensor): Log probability of encoders outputs will be passed to CTC Loss.
                If joint_ctc_attention is False, return None.
            **output_lengths** (torch.LongTensor): Tensor representing the length of the encoder output
                `LongTensor` of size ``(batch)``
        """
        encoder_logits = None

        output = inputs.transpose(1, 2)
        output_lengths = input_lengths

        for block in self.blocks:
            output, output_lengths = block(output, output_lengths)

        output = output.transpose(1, 2)

        if self.joint_ctc_attention:
            encoder_logits = self.fc(output).log_softmax(dim=2)

        return output, encoder_logits, output_lengths
