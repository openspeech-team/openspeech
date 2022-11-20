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

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from ...tokenizers.tokenizer import Tokenizer
from .. import register_criterion
from ..ctc.configuration import CTCLossConfigs


@register_criterion("ctc", dataclass=CTCLossConfigs)
class CTCLoss(nn.Module):
    r"""
    The Connectionist Temporal Classification loss.

    Calculates loss between a continuous (unsegmented) time series and a target sequence. CTCLoss sums over the
    probability of possible alignments of input to target, producing a loss value which is differentiable
    with respect to each input node. The alignment of input to target is assumed to be "many-to-one", which
    limits the length of the target sequence such that it must be :math:`\leq` the input length.

    Args:
        configs (DictConfig): hydra configuration set
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs: log_probs, targets, input_lengths, target_lengths
        - Log_probs: Tensor of size :math:`(T, N, C)`,
          where :math:`T = \text{input length}`,
          :math:`N = \text{batch size}`, and
          :math:`C = \text{number of classes (including blank)}`.
          The logarithmized probabilities of the outputs (e.g. obtained with
          :func:`torch.nn.functional.log_softmax`).
        - Targets: Tensor of size :math:`(N, S)` or
          :math:`(\operatorname{sum}(\text{target\_lengths}))`,
          where :math:`N = \text{batch size}` and
          :math:`S = \text{max target length, if shape is } (N, S)`.
          It represent the target sequences. Each element in the target
          sequence is a class index. And the target index cannot be blank (default=0).
          In the :math:`(N, S)` form, targets are padded to the
          length of the longest sequence, and stacked.
          In the :math:`(\operatorname{sum}(\text{target\_lengths}))` form,
          the targets are assumed to be un-padded and
          concatenated within 1 dimension.
        - Input_lengths: Tuple or tensor of size :math:`(N)`,
          where :math:`N = \text{batch size}`. It represent the lengths of the
          inputs (must each be :math:`\leq T`). And the lengths are specified
          for each sequence to achieve masking under the assumption that sequences
          are padded to equal lengths.
        - Target_lengths: Tuple or tensor of size :math:`(N)`,
          where :math:`N = \text{batch size}`. It represent lengths of the targets.
          Lengths are specified for each sequence to achieve masking under the
          assumption that sequences are padded to equal lengths. If target shape is
          :math:`(N,S)`, target_lengths are effectively the stop index
          :math:`s_n` for each target sequence, such that ``target_n = targets[n,0:s_n]`` for
          each target in a batch. Lengths must each be :math:`\leq S`
          If the targets are given as a 1d tensor that is the concatenation of individual
          targets, the target_lengths must add up to the total length of the tensor.

    Returns: loss
        * loss (float): loss for training

    Examples::

        >>> # Target are to be padded
        >>> T = 50      # Input sequence length
        >>> C = 20      # Number of classes (including blank)
        >>> N = 16      # Batch size
        >>> S = 30      # Target sequence length of longest target in batch (padding length)
        >>> S_min = 10  # Minimum target length, for demonstration purposes
        >>>
        >>> # Initialize random batch of input vectors, for *size = (T,N,C)
        >>> input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
        >>>
        >>> # Initialize random batch of targets (0 = blank, 1:C = classes)
        >>> target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
        >>>
        >>> input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        >>> target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
        >>> ctc_loss = nn.CTCLoss()
        >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
        >>> loss.backward()
        >>>
        >>>
        >>> # Target are to be un-padded
        >>> T = 50      # Input sequence length
        >>> C = 20      # Number of classes (including blank)
        >>> N = 16      # Batch size
        >>>
        >>> # Initialize random batch of input vectors, for *size = (T,N,C)
        >>> input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
        >>> input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
        >>>
        >>> # Initialize random batch of targets (0 = blank, 1:C = classes)
        >>> target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
        >>> target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)
        >>> ctc_loss = CTCLoss()
        >>> loss = ctc_loss(input, target, input_lengths, target_lengths)
        >>> loss.backward()

    Reference:
        A. Graves et al.: Connectionist Temporal Classification:
        Labelling Unsegmented Sequence Data with Recurrent Neural Networks:
        https://www.cs.toronto.edu/~graves/icml_2006.pdf
    """

    def __init__(
        self,
        configs: DictConfig,
        tokenizer: Tokenizer,
    ) -> None:
        super(CTCLoss, self).__init__()
        self.ctc_loss = nn.CTCLoss(
            blank=tokenizer.blank_id,
            reduction=configs.criterion.reduction,
            zero_infinity=configs.criterion.zero_infinity,
        )

    def forward(
        self,
        log_probs: Tensor,
        input_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ):
        return self.ctc_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )
