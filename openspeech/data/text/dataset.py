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

import torch
import logging

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    def __init__(self, transcripts: list, vocab):
        super(TextDataset, self).__init__()
        self.transcripts = transcripts
        self.vocab = vocab
        self.sos_id = vocab.sos_id
        self.eos_id = vocab.eos_id

    def _get_inputs(self, transcript):
        tokens = transcript.split(' ')
        transcript = [int(self.sos_id)]

        for token in tokens:
            transcript.append(int(token))

        return transcript

    def _get_targets(self, transcript):
        tokens = transcript.split(' ')
        transcript = list()

        for token in tokens:
            transcript.append(int(token))

        transcript.append(int(self.eos_id))

        return transcript

    def __getitem__(self, idx):
        transcript = self.vocab.string_to_label(self.transcripts[idx])
        inputs = torch.LongTensor(self._get_inputs(transcript))
        targets = torch.LongTensor(self._get_targets(transcript))
        return inputs, targets

    def __len__(self):
        return len(self.transcripts)

    def count(self):
        return len(self.transcripts)

