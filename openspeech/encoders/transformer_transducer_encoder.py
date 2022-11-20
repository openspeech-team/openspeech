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

import torch
import torch.nn as nn
from torch import Tensor

from openspeech.encoders import OpenspeechEncoder
from openspeech.modules import MultiHeadAttention, PositionalEncoding, PositionwiseFeedForward, get_attn_pad_mask


class TransformerTransducerEncoderLayer(nn.Module):
    r"""
    Repeated layers common to audio encoders and label encoders

    Args:
        model_dim (int): the number of features in the encoder (default : 512)
        d_ff (int): the number of features in the feed forward layers (default : 2048)
        num_heads (int): the number of heads in the multi-head attention (default: 8)
        dropout (float): dropout probability of encoder layer (default: 0.1)

    Inputs: inputs, self_attn_mask
        - **inputs**: Audio feature or label feature
        - **self_attn_mask**: Self attention mask to use in multi-head attention

    Returns: outputs, attn_distribution
        (Tensor, Tensor)

        * outputs (torch.FloatTensor): Tensor containing higher (audio, label) feature values
        * attn_distribution (torch.FloatTensor): Attention distribution in multi-head attention
    """

    def __init__(
        self,
        model_dim: int = 512,
        d_ff: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super(TransformerTransducerEncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(model_dim)
        self.self_attention = MultiHeadAttention(model_dim, num_heads)
        self.encoder_dropout = nn.Dropout(p=dropout)
        self.feed_forward = PositionwiseFeedForward(model_dim, d_ff, dropout)

    def forward(self, inputs: Tensor, self_attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        r"""
        Forward propagate a `inputs` for label encoder.

        Args:
            inputs : A input sequence passed to encoder layer. ``(batch, seq_length, dimension)``
            self_attn_mask : Self attention mask to cover up padding ``(batch, seq_length, seq_length)``

        Returns:
            **outputs** (Tensor): ``(batch, seq_length, dimension)``
            **attn_distribution** (Tensor): ``(batch, seq_length, seq_length)``
        """
        inputs = self.layer_norm(inputs)
        self_attn_output, attn_distribution = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        self_attn_output += inputs

        self_attn_output = self.layer_norm(self_attn_output)
        ff_output = self.feed_forward(self_attn_output)
        output = self.encoder_dropout(ff_output + self_attn_output)

        return output, attn_distribution


class TransformerTransducerEncoder(OpenspeechEncoder):
    r"""
    Converts the audio signal to higher feature values

    Args:
        input_size (int): dimension of input vector (default : 80)
        model_dim (int): the number of features in the audio encoder (default : 512)
        d_ff (int): the number of features in the feed forward layers (default : 2048)
        num_layers (int): the number of audio encoder layers (default: 18)
        num_heads (int): the number of heads in the multi-head attention (default: 8)
        dropout (float): dropout probability of audio encoder (default: 0.1)
        max_positional_length (int): Maximum length to use for positional encoding (default : 5000)

    Inputs: inputs, inputs_lens
        - **inputs**: Parsed audio of batch size number
        - **inputs_lens**: Tensor of sequence lengths

    Returns:
        * outputs (torch.FloatTensor): ``(batch, seq_length, dimension)``
        * input_lengths (torch.LongTensor):  ``(batch)``

    Reference:
        Qian Zhang et al.: Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss
        https://arxiv.org/abs/2002.02562
    """

    def __init__(
        self,
        input_size: int = 80,
        model_dim: int = 512,
        d_ff: int = 2048,
        num_layers: int = 18,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_positional_length: int = 5000,
    ) -> None:
        super(TransformerTransducerEncoder, self).__init__()
        self.input_size = input_size
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_positional_length)
        self.input_fc = nn.Linear(input_size, model_dim)
        self.encoder_layers = nn.ModuleList(
            [TransformerTransducerEncoderLayer(model_dim, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward propagate a `inputs` for audio encoder.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to audio encoder. Typically inputs will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            **outputs** (Tensor): ``(batch, seq_length, dimension)``
            ** input_lengths**(Tensor):  ``(batch)``
        """
        seq_len = inputs.size(1)

        self_attn_mask = get_attn_pad_mask(inputs, input_lengths, seq_len)

        inputs = self.input_fc(inputs) + self.positional_encoding(seq_len)
        outputs = self.input_dropout(inputs)

        for encoder_layer in self.encoder_layers:
            outputs, _ = encoder_layer(outputs, self_attn_mask)

        return outputs, input_lengths
