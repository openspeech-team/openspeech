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
from omegaconf import DictConfig
from torch import Tensor

from ...tokenizers.tokenizer import Tokenizer
from .. import register_criterion
from ..joint_ctc_cross_entropy.configuration import JointCTCCrossEntropyLossConfigs
from ..label_smoothed_cross_entropy.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyLoss


@register_criterion("joint_ctc_cross_entropy", dataclass=JointCTCCrossEntropyLossConfigs)
class JointCTCCrossEntropyLoss(nn.Module):
    r"""
    Privides Joint CTC-CrossEntropy Loss function. The logit from the encoder applies CTC Loss, and the logit
    from the decoder applies Cross Entropy. This loss makes the encoder more robust.

    Args:
        configs (DictConfig): hydra configuration set
        num_classes (int): the number of classfication
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs: encoder_logits, logits, output_lengths, targets, target_lengths
        - encoder_logits (torch.FloatTensor): probability distribution value from encoder and it has a logarithm shape.
            The `FloatTensor` of size ``(input_length, batch, num_classes)``
        - logits (torch.FloatTensor): probability distribution value from model and it has a logarithm shape.
            The `FloatTensor` of size ``(batch, seq_length, num_classes)``
        - output_lengths (torch.LongTensor): length of model's outputs.
            The `LongTensor` of size ``(batch)``
        - targets (torch.LongTensor): ground-truth encoded to integers which directly point a word in label.
            The `LongTensor` of size ``(batch, target_length)``
        - target_lengths (torch.LongTensor): length of targets.
            The `LongTensor` of size ``(batch)``

    Returns: loss, ctc_loss, cross_entropy_loss
        - loss (float): loss for training
        - ctc_loss (float): ctc loss for training
        - cross_entropy_loss (float): cross entropy loss for training

    Reference:
        Suyoun Kim et al.: Joint CTC-Attention based End-to-End Speech Recognition using Multi-task Learning:
        https://arxiv.org/abs/1609.06773
    """

    def __init__(
        self,
        configs: DictConfig,
        num_classes: int,
        tokenizer: Tokenizer,
    ) -> None:
        super(JointCTCCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.dim = -1
        self.ignore_index = tokenizer.pad_id
        self.reduction = configs.criterion.reduction.lower()
        self.ctc_weight = configs.criterion.ctc_weight
        self.cross_entropy_weight = configs.criterion.cross_entropy_weight
        self.ctc_loss = nn.CTCLoss(
            blank=tokenizer.blank_id,
            reduction=self.reduction,
            zero_infinity=configs.criterion.zero_infinity,
        )
        if configs.criterion.smoothing > 0.0:
            self.cross_entropy_loss = LabelSmoothedCrossEntropyLoss(
                configs=configs,
                num_classes=num_classes,
                tokenizer=tokenizer,
            )
        else:
            self.cross_entropy_loss = nn.CrossEntropyLoss(reduction=self.reduction, ignore_index=self.ignore_index)

    def forward(
        self,
        encoder_logits: Tensor,
        logits: Tensor,
        output_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        max_target_length = targets.size(1)
        max_logits_length = logits.size(1)

        if max_logits_length > max_target_length:
            logits = logits[:, :max_target_length, :]
            cross_entropy_targets = targets.clone()
        elif max_target_length > max_logits_length:
            cross_entropy_targets = targets[:, :max_logits_length].clone()
        else:
            cross_entropy_targets = targets.clone()

        logits = logits.contiguous().view(-1, logits.size(-1))

        ctc_loss = self.ctc_loss(encoder_logits, targets, output_lengths, target_lengths)
        cross_entropy_loss = self.cross_entropy_loss(logits, cross_entropy_targets.contiguous().view(-1))
        loss = cross_entropy_loss * self.cross_entropy_weight + ctc_loss * self.ctc_weight
        return loss, ctc_loss, cross_entropy_loss
