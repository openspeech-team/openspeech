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

import importlib
import os

CRITERION_REGISTRY = dict()
CRITERION_DATACLASS_REGISTRY = dict()


def register_criterion(name: str, dataclass=None):
    r"""
    New criterion types can be added to OpenSpeech with the :func:`register_criterion` function decorator.

    For example::
        @register_criterion('label_smoothed_cross_entropy')
        class LabelSmoothedCrossEntropyLoss(nn.Module):
            (...)

    .. note:: All criterion must implement the :class:`cls.__name__` interface.

    Args:
        name (str): the name of the criterion
        dataclass (Optional, str): the dataclass of the criterion (default: None)
    """

    def register_criterion_cls(cls):
        if name in CRITERION_REGISTRY:
            raise ValueError(f"Cannot register duplicate criterion ({name})")

        CRITERION_REGISTRY[name] = cls

        cls.__dataclass = dataclass
        if dataclass is not None:
            if name in CRITERION_DATACLASS_REGISTRY:
                raise ValueError(f"Cannot register duplicate criterion ({name})")
            CRITERION_DATACLASS_REGISTRY[name] = dataclass

        return cls

    return register_criterion_cls


criterion_dir = os.path.dirname(__file__)
for file in os.listdir(criterion_dir):
    if os.path.isdir(os.path.join(criterion_dir, file)) and not file.startswith("__"):
        for subfile in os.listdir(os.path.join(criterion_dir, file)):
            path = os.path.join(criterion_dir, file, subfile)
            if subfile.endswith(".py"):
                python_file = subfile[: subfile.find(".py")] if subfile.endswith(".py") else subfile
                module = importlib.import_module(f"openspeech.criterion.{file}.{python_file}")
        continue

    path = os.path.join(criterion_dir, file)
    if file.endswith(".py"):
        criterion_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(f"openspeech.criterion.{criterion_name}")


from .cross_entropy.configuration import CrossEntropyLossConfigs
from .cross_entropy.cross_entropy import CrossEntropyLoss
from .ctc.configuration import CTCLossConfigs
from .ctc.ctc import CTCLoss
from .joint_ctc_cross_entropy.configuration import JointCTCCrossEntropyLossConfigs
from .joint_ctc_cross_entropy.joint_ctc_cross_entropy import JointCTCCrossEntropyLoss
from .label_smoothed_cross_entropy.configuration import LabelSmoothedCrossEntropyLossConfigs
from .label_smoothed_cross_entropy.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyLoss
from .perplexity.perplexity import Perplexity, PerplexityLossConfigs
from .transducer.configuration import TransducerLossConfigs
from .transducer.transducer import TransducerLoss

__all__ = [
    "CrossEntropyLossConfigs",
    "CTCLossConfigs",
    "JointCTCCrossEntropyLossConfigs",
    "LabelSmoothedCrossEntropyLossConfigs",
    "TransducerLossConfigs",
    "CrossEntropyLoss",
    "CTCLoss",
    "JointCTCCrossEntropyLoss",
    "LabelSmoothedCrossEntropyLoss",
    "TransducerLoss",
]
