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

from typing import Union

import torch
import torch.nn as nn


class EnsembleSearch(nn.Module):
    """
    Class for ensemble search.

    Args:
        models (tuple): list of ensemble model

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be
            a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        * predictions (torch.LongTensor): prediction of ensemble models
    """

    def __init__(self, models: Union[list, tuple]):
        super(EnsembleSearch, self).__init__()
        assert len(models) > 1, "Ensemble search should be multiple models."
        self.models = models

    def forward(self, inputs: torch.FloatTensor, input_lengths: torch.LongTensor):
        logits = list()

        for model in self.models:
            output = model(inputs, input_lengths)
            logits.append(output["logits"])

        output = logits[0]

        for logit in logits[1:]:
            output += logit

        return output.max(-1)[1]


class WeightedEnsembleSearch(nn.Module):
    """
    Args:
        models (tuple): list of ensemble model
        weights (tuple: list of ensemble's weight

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be
            a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        * predictions (torch.LongTensor): prediction of ensemble models
    """

    def __init__(self, models: Union[list, tuple], weights: Union[list, tuple]):
        super(WeightedEnsembleSearch, self).__init__()
        assert len(models) > 1, "Ensemble search should be multiple models."
        assert len(models) == len(weights), "len(models), len(weight) should be same."
        self.models = models
        self.weights = weights

    def forward(self, inputs: torch.FloatTensor, input_lengths: torch.LongTensor):
        logits = list()

        for model in self.models:
            output = model(inputs, input_lengths)
            logits.append(output["logits"])

        output = logits[0] * self.weights[0]

        for idx, logit in enumerate(logits[1:]):
            output += logit * self.weights[1]

        return output.max(-1)[1]
