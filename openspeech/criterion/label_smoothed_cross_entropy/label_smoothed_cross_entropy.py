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
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor

from ...tokenizers.tokenizer import Tokenizer
from .. import register_criterion
from ..label_smoothed_cross_entropy.configuration import LabelSmoothedCrossEntropyLossConfigs


@register_criterion("label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyLossConfigs)
class LabelSmoothedCrossEntropyLoss(nn.Module):
    r"""
    Label smoothed cross entropy loss function.

    Args:
        configs (DictConfig): hydra configuration set
        num_classes (int): the number of classfication
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs: logits, targets
        - **logits** (torch.FloatTensor): probability distribution value from model and it has a logarithm shape.
            The `FloatTensor` of size ``(batch, seq_length, num_classes)``
        - **targets** (torch.LongTensor): ground-truth encoded to integers which directly point a word in label
            The `LongTensor` of size ``(batch, target_length)``

    Returns: loss
        * loss (float): loss for training
    """

    def __init__(
        self,
        configs: DictConfig,
        num_classes: int,
        tokenizer: Tokenizer,
    ) -> None:
        super(LabelSmoothedCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - configs.criterion.smoothing
        self.smoothing = configs.criterion.smoothing
        self.num_classes = num_classes
        self.dim = -1
        self.ignore_index = tokenizer.pad_id
        self.reduction = configs.criterion.reduction.lower()

        if self.reduction == "sum":
            self.reduction_method = torch.sum
        elif self.reduction == "mean":
            self.reduction_method = torch.mean
        else:
            raise ValueError(f"Unsupported reduction method {configs.criterion.reduction}")

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # If predict longer than the target size, won't be able to calculate the cross entropy
        max_target_length = targets.size(1)
        max_logits_length = logits.size(1)

        if max_logits_length > max_target_length:
            logits = logits[:, :max_target_length, :]
        elif max_target_length > max_logits_length:
            targets = targets[:, :max_logits_length]

        logits = logits.contiguous().view(-1, logits.size(-1))
        targets = targets.contiguous().view(-1)

        if self.smoothing > 0.0:
            with torch.no_grad():
                label_smoothed = torch.zeros_like(logits)
                label_smoothed.fill_(self.smoothing / (self.num_classes - 1))
                label_smoothed.scatter_(1, targets.data.unsqueeze(1), self.confidence)
                label_smoothed[targets == self.ignore_index, :] = 0
            return self.reduction_method(-label_smoothed * logits)

        return F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction=self.reduction)
