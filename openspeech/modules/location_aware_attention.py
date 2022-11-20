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
from torch import Tensor

from openspeech.modules.wrapper import Linear


class LocationAwareAttention(nn.Module):
    r"""
    Applies a location-aware attention mechanism on the output features from the decoders.
    Location-aware attention proposed in "Attention-Based Models for Speech Recognition" paper.
    The location-aware attention mechanism is performing well in speech recognition tasks.
    We refer to implementation of ClovaCall Attention style.

    Args:
        dim (int): dimension of model
        attn_dim (int): dimension of attention
        smoothing (bool): flag indication whether to use smoothing or not.

    Inputs: query, value, last_attn
        - **query** (batch, q_len, hidden_dim): tensor containing the output features from the decoders.
        - **value** (batch, v_len, hidden_dim): tensor containing features of the encoded input sequence.
        - **last_attn** (batch_size, v_len): tensor containing previous timestep`s attention (alignment)

    Returns: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the feature from encoders outputs
        - **attn** (batch * num_heads, v_len): tensor containing the attention (alignment) from the encoders outputs.

    Reference:
        Jan Chorowski et al.: Attention-Based Models for Speech Recognition.
        https://arxiv.org/abs/1506.07503
    """

    def __init__(self, dim: int = 1024, attn_dim: int = 1024, smoothing: bool = False) -> None:
        super(LocationAwareAttention, self).__init__()
        self.location_conv = nn.Conv1d(in_channels=1, out_channels=attn_dim, kernel_size=3, padding=1)
        self.query_proj = Linear(dim, attn_dim, bias=False)
        self.value_proj = Linear(dim, attn_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(attn_dim).uniform_(-0.1, 0.1))
        self.fc = Linear(attn_dim, 1, bias=True)
        self.smoothing = smoothing

    def forward(self, query: Tensor, value: Tensor, last_alignment_energy: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, seq_length = query.size(0), query.size(2), value.size(1)

        if last_alignment_energy is None:
            last_alignment_energy = value.new_zeros(batch_size, seq_length)

        last_alignment_energy = self.location_conv(last_alignment_energy.unsqueeze(dim=1))
        last_alignment_energy = last_alignment_energy.transpose(1, 2)

        alignmment_energy = self.fc(
            torch.tanh(self.query_proj(query) + self.value_proj(value) + last_alignment_energy + self.bias)
        ).squeeze(dim=-1)

        if self.smoothing:
            alignmment_energy = torch.sigmoid(alignmment_energy)
            alignmment_energy = torch.div(alignmment_energy, alignmment_energy.sum(dim=-1).unsqueeze(dim=-1))

        else:
            alignmment_energy = F.softmax(alignmment_energy, dim=-1)

        context = torch.bmm(alignmment_energy.unsqueeze(dim=1), value)

        return context, alignmment_energy
