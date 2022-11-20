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

import numpy as np
from omegaconf import DictConfig
from torch import Tensor

from ....utils import TORCHAUDIO_IMPORT_ERROR
from ... import register_audio_feature_transform
from ...audio.filter_bank.configuration import FilterBankConfigs


@register_audio_feature_transform("fbank", dataclass=FilterBankConfigs)
class FilterBankFeatureTransform(object):
    r"""
    Create a fbank from a raw audio signal. This matches the input/output of Kaldi's compute-fbank-feats.

    Args:
        configs (DictConfig): hydra configuraion set

    Inputs:
        signal (np.ndarray): signal from audio file.

    Returns:
        Tensor: A fbank identical to what Kaldi would output. The shape is ``(seq_length, num_mels)``
    """

    def __init__(self, configs: DictConfig) -> None:
        super(FilterBankFeatureTransform, self).__init__()
        try:
            import torchaudio
        except ImportError:
            raise ImportError(TORCHAUDIO_IMPORT_ERROR)
        self.num_mels = configs.audio.num_mels
        self.frame_length = configs.audio.frame_length
        self.frame_shift = configs.audio.frame_shift
        self.function = torchaudio.compliance.kaldi.fbank

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Provides feature extraction

        Inputs:
            signal (np.ndarray): audio signal

        Returns:
            feature (np.ndarray): feature extract by sub-class
        """
        return (
            self.function(
                Tensor(signal).unsqueeze(0),
                num_mel_bins=self.num_mels,
                frame_length=self.frame_length,
                frame_shift=self.frame_shift,
            )
            .transpose(0, 1)
            .numpy()
        )
