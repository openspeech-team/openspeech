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

from ...tokenizers.tokenizer import Tokenizer
from ...utils import WARPRNNT_IMPORT_ERROR
from .. import register_criterion
from ..transducer.configuration import TransducerLossConfigs


@register_criterion("transducer", dataclass=TransducerLossConfigs)
class TransducerLoss(nn.Module):
    r"""
    Compute path-aware regularization transducer loss.

    Args:
        configs (DictConfig): hydra configuration set
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        logits (torch.FloatTensor): Input tensor with shape (N, T, U, V)
            where N is the minibatch size, T is the maximum number of
            input frames, U is the maximum number of output labels and V is
            the vocabulary of labels (including the blank).
        targets (torch.IntTensor): Tensor with shape (N, U-1) representing the
            reference labels for all samples in the minibatch.
        input_lengths (torch.IntTensor): Tensor with shape (N,) representing the
            number of frames for each sample in the minibatch.
        target_lengths (torch.IntTensor): Tensor with shape (N,) representing the
            length of the transcription for each sample in the minibatch.

    Returns:
        - loss (torch.FloatTensor): transducer loss

    Reference:
        A. Graves: Sequence Transduction with Recurrent Neural Networks:
        https://arxiv.org/abs/1211.3711.pdf
    """

    def __init__(
        self,
        configs: DictConfig,
        tokenizer: Tokenizer,
    ) -> None:
        super().__init__()
        try:
            from warp_rnnt import rnnt_loss
        except ImportError:
            raise ImportError(WARPRNNT_IMPORT_ERROR)
        self.rnnt_loss = rnnt_loss
        self.blank_id = tokenizer.blank_id
        self.reduction = configs.criterion.reduction
        self.gather = configs.criterion.gather

    def forward(
        self,
        logits: torch.FloatTensor,
        targets: torch.IntTensor,
        input_lengths: torch.IntTensor,
        target_lengths: torch.IntTensor,
    ) -> torch.FloatTensor:
        return self.rnnt_loss(
            logits,
            targets,
            input_lengths,
            target_lengths,
            reduction=self.reduction,
            blank=self.blank_id,
            gather=self.gather,
        )
