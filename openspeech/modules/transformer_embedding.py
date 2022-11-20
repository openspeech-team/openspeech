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

import math

import torch.nn as nn
from torch import Tensor


class TransformerEmbedding(nn.Module):
    r"""
    Embedding layer. Similarly to other sequence transduction models, transformer use learned embeddings
    to convert the input tokens and output tokens to vectors of dimension d_model.
    In the embedding layers, transformer multiply those weights by sqrt(d_model)

    Args:
        num_embeddings (int): the number of embedding size
        pad_id (int): identification of pad token
        d_model (int): dimension of model

    Inputs:
        inputs (torch.FloatTensor): input of embedding layer

    Returns:
        outputs (torch.FloatTensor): output of embedding layer
    """

    def __init__(self, num_embeddings: int, pad_id: int, d_model: int = 512) -> None:
        super(TransformerEmbedding, self).__init__()
        self.sqrt_dim = math.sqrt(d_model)
        self.embedding = nn.Embedding(num_embeddings, d_model, padding_idx=pad_id)

    def forward(self, inputs: Tensor) -> Tensor:
        r"""
        Forward propagate of embedding layer.

        Inputs:
            inputs (torch.FloatTensor): input of embedding layer

        Returns:
            outputs (torch.FloatTensor): output of embedding layer
        """
        return self.embedding(inputs) * self.sqrt_dim
