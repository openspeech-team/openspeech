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

import os

import numpy as np
from torch.utils.data import Sampler

from .audio.load import load_audio


class RandomSampler(Sampler):
    r"""
    Implementation of a Random Sampler for sampling the dataset.

    Args:
        data_source (torch.utils.data.Dataset): dataset to sample from
        batch_size (int): size of batch
        drop_last (bool): flat indication whether to drop last batch or not
    """

    def __init__(self, data_source, batch_size: int = 32, drop_last: bool = False) -> None:
        super(RandomSampler, self).__init__(data_source)
        self.batch_size = batch_size
        self.data_source = data_source
        ids = list(range(0, len(data_source)))
        self.bins = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]
        self.drop_last = drop_last

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)


class SmartBatchingSampler(Sampler):
    """
    Batching with similar sequence length.

    Args:
        data_source (torch.utils.data.Dataset): dataset to sample from
        batch_size (int): size of batch
        drop_last (bool): flat indication whether to drop last batch or not
    """

    def __init__(self, data_source, batch_size: int = 32, drop_last: bool = False) -> None:
        super(SmartBatchingSampler, self).__init__(data_source)
        self.batch_size = batch_size
        self.data_source = data_source

        audio_lengths = [self._get_audio_length(audio_path) for audio_path in data_source.audio_paths]
        audio_indices = [idx for idx in range(len(data_source.audio_paths))]

        pack_by_length = list(zip(audio_lengths, audio_indices))
        sort_by_length = sorted(pack_by_length)
        audio_lengths, audio_indices = zip(*sort_by_length)

        self.bins = [audio_indices[i : i + batch_size] for i in range(0, len(audio_indices), batch_size)]
        self.drop_last = drop_last

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(list(ids))
            yield ids

    def _get_audio_length(self, audio_path):
        return len(load_audio(os.path.join(self.data_source.dataset_path, audio_path), sample_rate=16000))

    def __len__(self):
        return len(self.bins)

    def shuffle(self, epoch):
        np.random.shuffle(self.bins)
