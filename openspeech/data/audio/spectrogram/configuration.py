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

from ....dataclass.configurations import OpenspeechDataclass


@dataclass
class SpectrogramConfigs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.data.audio.SpectrogramTransform`.

    It is used to initiated an `SpectrogramTransform` feature transform.

    Configuration objects inherit from :class: `~openspeech.dataclass.OpenspeechDataclass`.

    Args:
        name (str): name of feature transform. (default: spectrogram)
        sample_rate (int): sampling rate of audio (default: 16000)
        frame_length (float): frame length for spectrogram (default: 20.0)
        frame_shift (float): length of hop between STFT (default: 10.0)
        del_silence (bool): flag indication whether to apply delete silence or not (default: False)
        num_mels (int): the number of mfc coefficients to retain. (default: 161)
        apply_spec_augment (bool): flag indication whether to apply spec augment or not (default: True)
        apply_noise_augment (bool): flag indication whether to apply noise augment or not (default: False)
        apply_time_stretch_augment (bool): flag indication whether to apply time stretch augment or not (default: False)
        apply_joining_augment (bool): flag indication whether to apply audio joining augment or not (default: False)
    """
    name: str = field(default="spectrogram", metadata={"help": "Name of dataset."})
    sample_rate: int = field(default=16000, metadata={"help": "Sampling rate of audio"})
    frame_length: float = field(default=20.0, metadata={"help": "Frame length for spectrogram"})
    frame_shift: float = field(default=10.0, metadata={"help": "Length of hop between STFT"})
    del_silence: bool = field(
        default=False, metadata={"help": "Flag indication whether to apply delete silence or not"}
    )
    num_mels: int = field(
        default=161,
        metadata={
            "help": "Spectrogram is independent of mel, but uses the 'num_mels' variable "
            "to unify feature size variables "
        },
    )
    apply_spec_augment: bool = field(
        default=True, metadata={"help": "Flag indication whether to apply spec augment or not"}
    )
    apply_noise_augment: bool = field(
        default=False, metadata={"help": "Flag indication whether to apply noise augment or not"}
    )
    apply_time_stretch_augment: bool = field(
        default=False, metadata={"help": "Flag indication whether to apply time stretch augment or not"}
    )
    apply_joining_augment: bool = field(
        default=False, metadata={"help": "Flag indication whether to apply audio joining augment or not"}
    )
