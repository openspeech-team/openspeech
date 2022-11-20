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

from openspeech.lm.openspeech_lm import OpenspeechLanguageModelBase
from openspeech.modules import (
    Linear,
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    TransformerEmbedding,
    get_attn_pad_mask,
    get_attn_subsequent_mask,
)


class TransformerForLanguageModelLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        num_attention_heads: int = 8,
        d_ff: int = 2048,
        dropout_p: float = 0.3,
    ) -> None:
        super(TransformerForLanguageModelLayer, self).__init__()
        self.attention_prenorm = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_attention_heads)
        self.feed_forward_prenorm = nn.LayerNorm(d_model)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout_p=dropout_p)

    def forward(
        self,
        inputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = inputs
        inputs = self.attention_prenorm(inputs)
        outputs, _ = self.attention(inputs, inputs, inputs, mask)
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_prenorm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual

        return outputs


class TransformerForLanguageModel(OpenspeechLanguageModelBase):
    """
    Language Modelling is the core problem for a number of of natural language processing tasks such as speech to text,
    conversational system, and text summarization. A trained language model learns the likelihood of occurrence
    of a word based on the previous sequence of words used in the text.

    Args:
        num_classes (int): number of classification
        max_length (int): max decoding length (default: 128)
        d_model (int): dimension of model (default: 768)
        d_ff (int): dimension of feed forward network (default: 1536)
        num_attention_heads (int): number of attention heads (default: 8)
        pad_id (int, optional): index of the pad symbol (default: 0)
        sos_id (int, optional): index of the start of sentence symbol (default: 1)
        eos_id (int, optional): index of the end of sentence symbol (default: 2)
        num_layers (int, optional): number of transformer layers (default: 2)
        dropout_p (float, optional): dropout probability of decoders (default: 0.2)

    Inputs:, inputs, input_lengths
        inputs (torch.LongTensor): A input sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        * logits (torch.FloatTensor): Log probability of model predictions.
    """

    def __init__(
        self,
        num_classes: int,
        max_length: int = 128,
        d_model: int = 768,
        num_attention_heads: int = 8,
        d_ff: int = 1536,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        num_layers: int = 2,
        dropout_p: float = 0.3,
    ):
        super(TransformerForLanguageModel, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.embedding = TransformerEmbedding(num_classes, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.layers = nn.ModuleList(
            [
                TransformerForLanguageModelLayer(
                    d_model=d_model,
                    num_attention_heads=num_attention_heads,
                    d_ff=d_ff,
                    dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            Linear(d_model, d_model, bias=False),
            nn.Tanh(),
            Linear(d_model, num_classes, bias=False),
        )

    def forward_step(self, inputs, input_lengths):
        pad_mask = get_attn_pad_mask(inputs, input_lengths, inputs.size(1))
        subsequent_mask = get_attn_subsequent_mask(inputs)
        mask = torch.gt((pad_mask + subsequent_mask), 0)

        outputs = self.embedding(inputs) + self.positional_encoding(inputs.size(1))
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs = layer(inputs=outputs, mask=mask)

        step_outputs = self.fc(outputs).log_softmax(dim=-1)

        return step_outputs

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward propagate a `encoder_outputs` for training.

        Args:
            inputs (torch.LongTensor): A input sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * logits (torch.FloatTensor): Log probability of model predictions.
        """
        logits = list()

        step_outputs = self.forward_step(inputs, input_lengths)

        for di in range(step_outputs.size(1)):
            step_output = step_outputs[:, di, :]
            logits.append(step_output)

        return torch.stack(logits, dim=1)
