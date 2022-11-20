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
import os
import random
from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig

from openspeech.data.sampler import RandomSampler
from openspeech.data.text.data_loader import TextDataLoader
from openspeech.data.text.dataset import TextDataset
from openspeech.datasets import register_data_module
from openspeech.tokenizers.tokenizer import Tokenizer


@register_data_module("lm")
class LightningLanguageModelDataModule(pl.LightningDataModule):
    def __init__(self, configs: DictConfig) -> None:
        super(LightningLanguageModelDataModule, self).__init__()
        self.configs = configs
        self.dataset = dict()
        self.logger = logging.getLogger(__name__)

    def prepare_data(self):
        if not os.path.exists(self.configs.dataset.dataset_path):
            raise FileNotFoundError

    def setup(self, stage: Optional[str] = None, tokenizer: Tokenizer = None):
        num_total_transcripts = 0
        transcripts = list()

        with open(self.configs.dataset.dataset_path, encoding=self.configs.tokenizer.encoding) as f:
            for line in f.readlines():
                transcripts.append(line)
                num_total_transcripts += 1

        random.shuffle(transcripts)

        train_ratio = 1 - self.configs.dataset.valid_ratio - self.configs.dataset.test_ratio

        num_train_transcripts = int(num_total_transcripts * train_ratio)
        num_valid_transcripts = int(num_total_transcripts * self.configs.dataset.valid_ratio)

        valid_end_idx = num_train_transcripts + num_valid_transcripts

        transcripts = {
            "train": transcripts[:num_train_transcripts],
            "valid": transcripts[num_train_transcripts:valid_end_idx],
            "test": transcripts[valid_end_idx:],
        }

        for stage in transcripts.keys():
            self.dataset[stage] = TextDataset(
                transcripts=transcripts[stage],
                tokenizer=tokenizer,
            )

    def train_dataloader(self) -> TextDataLoader:
        train_sampler = RandomSampler(self.dataset["train"], batch_size=self.configs.trainer.batch_size)
        return TextDataLoader(
            dataset=self.dataset["train"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=train_sampler,
        )

    def val_dataloader(self) -> TextDataLoader:
        r"""Return data loader for validation."""
        valid_sampler = RandomSampler(self.dataset["valid"], batch_size=self.configs.trainer.batch_size)
        return TextDataLoader(
            dataset=self.dataset["valid"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=valid_sampler,
        )

    def test_dataloader(self) -> TextDataLoader:
        r"""Return data loader for training."""
        train_sampler = RandomSampler(self.dataset["test"], batch_size=self.configs.trainer.batch_size)
        return TextDataLoader(
            dataset=self.dataset["test"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=train_sampler,
        )
