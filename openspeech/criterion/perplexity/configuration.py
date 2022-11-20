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

from dataclasses import dataclass, field

from ...dataclass.configurations import OpenspeechDataclass


@dataclass
class PerplexityLossConfigs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of a
    :class: `~openspeech.criterion.Perplexity`.

    It is used to initiated an `Perplexity` criterion.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Configurations:
        criterion_name (str): name of criterion (default: perplexity)
        reduction (str): reduction method of criterion (default: mean)
    """
    criterion_name: str = field(default="perplexity", metadata={"help": "Criterion name for training"})
    reduction: str = field(default="mean", metadata={"help": "Reduction method of criterion"})
