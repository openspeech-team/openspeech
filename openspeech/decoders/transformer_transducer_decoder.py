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

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from openspeech.decoders import OpenspeechDecoder
from openspeech.encoders.transformer_transducer_encoder import TransformerTransducerEncoderLayer
from openspeech.modules import (
    PositionalEncoding,
    get_attn_pad_mask,
    get_attn_subsequent_mask,
)


class TransformerTransducerDecoder(OpenspeechDecoder):
    r"""
    Converts the label to higher feature values

    Args:
        num_classes (int): the number of vocabulary
        model_dim (int): the number of features in the label encoder (default : 512)
        d_ff (int): the number of features in the feed forward layers (default : 2048)
        num_layers (int): the number of label encoder layers (default: 2)
        num_heads (int): the number of heads in the multi-head attention (default: 8)
        dropout (float): dropout probability of label encoder (default: 0.1)
        max_positional_length (int): Maximum length to use for positional encoding (default : 5000)
        pad_id (int): index of padding (default: 0)
        sos_id (int): index of the start of sentence (default: 1)
        eos_id (int): index of the end of sentence (default: 2)

    Inputs: inputs, inputs_lens
        - **inputs**: Ground truth of batch size number
        - **inputs_lens**: Tensor of target lengths

    Returns:
        (torch.FloatTensor, torch.FloatTensor)

        * outputs (torch.FloatTensor): ``(batch, seq_length, dimension)``
        * input_lengths (torch.FloatTensor):  ``(batch)``

    Reference:
        Qian Zhang et al.: Transformer Transducer: A Streamable Speech Recognition Model with Transformer Encoders and RNN-T Loss
        https://arxiv.org/abs/2002.02562
    """
    def __init__(
            self,
            num_classes: int,
            model_dim: int = 512,
            d_ff: int = 2048,
            num_layers: int = 2,
            num_heads: int = 8,
            dropout: float = 0.1,
            max_positional_length: int = 5000,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
    ) -> None:
        super(TransformerTransducerDecoder, self).__init__()
        self.embedding = nn.Embedding(num_classes, model_dim)
        self.scale = np.sqrt(model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_positional_length)
        self.input_dropout = nn.Dropout(p=dropout)
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.encoder_layers = nn.ModuleList([
            TransformerTransducerEncoderLayer(
                model_dim,
                d_ff,
                num_heads,
                dropout
            ) for _ in range(num_layers)
        ])

    def forward(
            self,
            inputs: torch.Tensor,
            input_lengths: torch.Tensor,
            hidden_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Forward propagate a `inputs` for label encoder.

        Args:
            inputs (torch.LongTensor): A input sequence passed to label encoder. Typically inputs will be a padded
                `LongTensor` of size ``(batch, target_length)``
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * outputs (Tensor): ``(batch, seq_length, dimension)``
            * input_lengths (Tensor):  ``(batch)``
        """
        self_attn_mask = None
        batch = inputs.size(0)

        inputs = inputs[inputs != self.eos_id].view(batch, -1)

        if len(inputs.size()) == 1:  # validate, evaluation
            inputs = inputs.unsqueeze(1)
            target_lens = inputs.size(1)

            embedding_output = self.embedding(inputs) * self.scale
            positional_encoding_output = self.positional_encoding(target_lens)
            inputs = embedding_output + positional_encoding_output

        else:  # train
            inputs = inputs[inputs != self.eos_id].view(batch, -1)
            target_lengths = inputs.size(1)

            embedding_output = self.embedding(inputs) * self.scale
            positional_encoding_output = self.positional_encoding(target_lengths)
            inputs = embedding_output + positional_encoding_output

            dec_self_attn_pad_mask = get_attn_pad_mask(inputs, input_lengths, inputs.size(1))
            dec_self_attn_subsequent_mask = get_attn_subsequent_mask(inputs)
            self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        outputs = self.input_dropout(inputs)

        for encoder_layer in self.encoder_layers:
            outputs, _ = encoder_layer(outputs, self_attn_mask)

        return outputs, input_lengths
