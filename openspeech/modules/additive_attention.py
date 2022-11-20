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


class AdditiveAttention(nn.Module):
    r"""
    Applies a additive attention (bahdanau) mechanism on the output features from the decoders.
    Additive attention proposed in "Neural Machine Translation by Jointly Learning to Align and Translate" paper.

    Args:
        dim (int): dimension of model

    Inputs: query, key, value
        - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoders.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoders.
        - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.

    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the alignment from the encoders outputs.
    """

    def __init__(self, dim: int) -> None:
        super(AdditiveAttention, self).__init__()
        self.query_proj = Linear(dim, dim, bias=False)
        self.key_proj = Linear(dim, dim, bias=False)
        self.score_proj = Linear(dim, 1)
        self.bias = nn.Parameter(torch.rand(dim).uniform_(-0.1, 0.1))

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)

        context += query

        return context, attn
