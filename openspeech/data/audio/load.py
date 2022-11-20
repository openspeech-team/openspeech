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

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def load_audio(audio_path: str, sample_rate: int, del_silence: bool = False) -> np.ndarray:
    """
    Load audio file (PCM) to sound. if del_silence is True, Eliminate all sounds below 30dB.
    If exception occurs in numpy.memmap(), return None.
    """
    try:
        if audio_path.endswith("pcm"):
            signal = np.memmap(audio_path, dtype="h", mode="r").astype("float32")

            if del_silence:
                non_silence_indices = librosa.effects.split(signal, top_db=30)
                signal = np.concatenate([signal[start:end] for start, end in non_silence_indices])

            return signal / 32767  # normalize audio

        elif audio_path.endswith("wav") or audio_path.endswith("flac"):
            signal, _ = librosa.load(audio_path, sr=sample_rate)
            return signal

    except ValueError:
        logger.warning("ValueError in {0}".format(audio_path))
        return None
    except RuntimeError:
        logger.warning("RuntimeError in {0}".format(audio_path))
        return None
    except IOError:
        logger.warning("IOError in {0}".format(audio_path))
        return None
