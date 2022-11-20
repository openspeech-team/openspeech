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
from omegaconf import DictConfig
from torch import Tensor

from ...tokenizers.tokenizer import Tokenizer
from .. import register_criterion
from ..perplexity.configuration import PerplexityLossConfigs


@register_criterion("perplexity", dataclass=PerplexityLossConfigs)
class Perplexity(nn.Module):
    r"""
    Language model perplexity loss.
    Perplexity is the token averaged likelihood.  When the averaging options are the
    same, it is the exponential of negative log-likelihood.

    Args:
        configs (DictConfig): hydra configuration set
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs: logits, targets
        - **logits** (torch.FloatTensor): probability distribution value from model and it has a logarithm shape.
            The `FloatTensor` of size ``(batch, seq_length, num_classes)``
        - **targets** (torch.LongTensor): ground-truth encoded to integers which directly point a word in label
            The `LongTensor` of size ``(batch, target_length)``

    Returns: loss
        - loss (float): loss for training
    """

    def __init__(
        self,
        configs: DictConfig,
        tokenizer: Tokenizer,
    ) -> None:
        super(Perplexity, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            reduction=configs.criterion.reduction,
            ignore_index=tokenizer.pad_id,
        )

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        max_target_length = targets.size(1)
        max_logits_length = logits.size(1)

        if max_logits_length > max_target_length:
            logits = logits[:, :max_target_length, :]
        elif max_target_length > max_logits_length:
            targets = targets[:, :max_logits_length]

        logits = logits.contiguous().view(-1, logits.size(-1))

        cross_entropy_loss = self.cross_entropy_loss(
            logits.contiguous().view(-1, logits.size(-1)),
            targets.contiguous().view(-1),
        )
        return torch.exp(cross_entropy_loss)
