import unittest

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from librosa.display import specshow

from openspeech.data.audio.augment import JoiningAugment, SpecAugment, TimeStretchAugment
from openspeech.utils import DUMMY_FEATURES, DUMMY_SIGNALS, DUMMY_TRANSCRIPTS


class TestAudioAugment(unittest.TestCase):
    def test_spec_augment(self):
        spec_augment = SpecAugment(freq_mask_para=10)
        specshow(librosa.power_to_db(DUMMY_FEATURES, ref=np.max), y_axis="mel", fmax=8000, x_axis="time")
        plt.title("Original Mel-Spectrogram")
        plt.tight_layout()
        plt.show()
        feature = spec_augment(DUMMY_FEATURES.transpose(0, 1))
        specshow(librosa.power_to_db(DUMMY_FEATURES, ref=np.max), y_axis="mel", fmax=8000, x_axis="time")
        plt.title("SpecAugmented Mel-Spectrogram")
        plt.tight_layout()
        plt.show()
        assert isinstance(feature, torch.Tensor)

    def test_time_stretch_augment(self):
        y, sr = librosa.load(librosa.ex("choice"))
        plt.plot(y)
        plt.title("Original Signal")
        plt.tight_layout()
        plt.show()

        slow_time_stretch = TimeStretchAugment(min_rate=0.4, max_rate=0.7)
        stretched_signal = slow_time_stretch(y)
        plt.plot(stretched_signal)
        plt.title("Slow Stretched Signal")
        plt.tight_layout()
        plt.show()

        fast_time_stretch = TimeStretchAugment(min_rate=1.4, max_rate=1.7)
        stretched_signal = fast_time_stretch(y)
        plt.plot(stretched_signal)
        plt.title("Fast Stretched Signal")
        plt.tight_layout()
        plt.show()

    def test_audio_joining(self):
        joining_augment = JoiningAugment()

        plt.plot(DUMMY_SIGNALS)
        plt.title(DUMMY_TRANSCRIPTS)
        plt.tight_layout()
        plt.show()

        joined_audio = joining_augment((DUMMY_SIGNALS, DUMMY_SIGNALS))
        plt.plot(joined_audio)
        plt.title(DUMMY_TRANSCRIPTS + DUMMY_TRANSCRIPTS)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    unittest.main()
