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

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler


def _collate_fn(batch, pad_id: int = 0):
    r"""
    Functions that pad to the maximum sequence length

    Args:
        batch (tuple): tuple contains input and target tensors
        pad_id (int): identification of pad token

    Returns:
        seqs (torch.FloatTensor): tensor contains input sequences.
        target (torch.IntTensor): tensor contains target sequences.
        seq_lengths (torch.IntTensor): tensor contains input sequence lengths
        target_lengths (torch.IntTensor): tensor contains target sequence lengths
    """

    def seq_length_(p):
        return len(p[0])

    def target_length_(p):
        return len(p[1])

    # sort by sequence length for rnn.pack_padded_sequence()
    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)

    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) - 1 for s in batch]

    max_seq_sample = max(batch, key=seq_length_)[0]
    max_target_sample = max(batch, key=target_length_)[1]

    max_seq_size = max_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)

    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(pad_id)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(0)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)
    target_lengths = torch.IntTensor(target_lengths)

    return seqs, targets, seq_lengths, target_lengths


class AudioDataLoader(DataLoader):
    r"""
    Audio Data Loader

    Args:
        dataset (torch.utils.data.Dataset): dataset from which to load the data.
        num_workers (int): how many subprocesses to use for data loading.
        batch_sampler (torch.utils.data.sampler.Sampler): defines the strategy to draw samples from the dataset.
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_workers: int,
        batch_sampler: torch.utils.data.sampler.Sampler,
        **kwargs,
    ) -> None:
        super(AudioDataLoader, self).__init__(
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            **kwargs,
        )
        self.collate_fn = _collate_fn


def load_dataset(manifest_file_path: str) -> Tuple[list, list]:
    """
    Provides dictionary of filename and labels.

    Args:
        manifest_file_path (str): evaluation manifest file path.

    Returns: target_dict
        * target_dict (dict): dictionary of filename and labels
    """
    audio_paths = list()
    transcripts = list()

    with open(manifest_file_path) as f:
        for idx, line in enumerate(f.readlines()):
            audio_path, korean_transcript, transcript = line.split("\t")
            transcript = transcript.replace("\n", "")

            audio_paths.append(audio_path)
            transcripts.append(transcript)

    return audio_paths, transcripts
