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

from ....utils import LIBROSA_IMPORT_ERROR
from ... import register_audio_feature_transform
from ...audio.mfcc.configuration import MFCCConfigs


@register_audio_feature_transform("mfcc", dataclass=MFCCConfigs)
class MFCCFeatureTransform(object):
    r"""
    Create the Mel-frequency cepstrum coefficients from an audio signal.

    By default, this calculates the MFCC on the DB-scaled Mel spectrogram.
    This is not the textbook implementation, but is implemented here to
    give consistency with librosa.

    This output depends on the maximum value in the input spectrogram, and so
    may return different values for an audio clip split into snippets vs. a
    a full clip.

    Args:
        configs (DictConfig): configuraion set

    Returns:
        Tensor: A mfcc feature. The shape is ``(seq_length, num_mels)``
    """

    def __init__(self, configs: DictConfig) -> None:
        super(MFCCFeatureTransform, self).__init__()
        try:
            import librosa
        except ImportError:
            raise ImportError(LIBROSA_IMPORT_ERROR)
        self.sample_rate = configs.audio.sample_rate
        self.num_mels = configs.audio.num_mels
        self.n_fft = int(round(configs.audio.sample_rate * 0.001 * configs.audio.frame_length))
        self.hop_length = int(round(configs.audio.sample_rate * 0.001 * configs.audio.frame_shift))
        self.function = librosa.feature.mfcc

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Provides feature extraction

        Inputs:
            signal (np.ndarray): audio signal

        Returns:
            feature (np.ndarray): feature extract by sub-class
        """
        return self.function(
            y=signal,
            sr=self.sample_rate,
            n_mfcc=self.num_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
