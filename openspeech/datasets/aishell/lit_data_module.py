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
import tarfile
from typing import Optional, Tuple

import pytorch_lightning as pl
import wget
from omegaconf import DictConfig

from openspeech.data.audio.data_loader import AudioDataLoader
from openspeech.data.audio.dataset import SpeechToTextDataset
from openspeech.data.sampler import RandomSampler, SmartBatchingSampler
from openspeech.datasets import register_data_module
from openspeech.datasets.aishell.preprocess import generate_character_labels, generate_character_script


@register_data_module("aishell")
class LightningAIShellDataModule(pl.LightningDataModule):
    r"""
    Lightning data module for AIShell-1. The corpus includes training set, development set and test sets.
    Training set contains 120,098 utterances from 340 speakers; development set contains 14,326 utterance from
    the 40 speakers; Test set contains 7,176 utterances from 20 speakers. For each speaker, around 360 utterances
    (about 26 minutes of speech) are released.

    Args:
        configs (DictConfig): configuration set.
    """
    AISHELL_TRAIN_NUM = 120098
    AISHELL_VALID_NUM = 14326
    AISHELL_TEST_NUM = 7176

    def __init__(self, configs: DictConfig):
        super(LightningAIShellDataModule, self).__init__()
        self.configs = configs
        self.dataset = dict()
        self.logger = logging.getLogger(__name__)

    def _download_dataset(self) -> None:
        r"""Download aishell dataset."""
        url = "https://www.openslr.org/resources/33/data_aishell.tgz"

        if not os.path.exists(self.configs.dataset.dataset_path):
            os.mkdir(self.configs.dataset.dataset_path)

        wget.download(url, f"{self.configs.dataset.dataset_path}/data_aishell.tgz")

        self.logger.info(f"Un-tarring archive {self.configs.dataset.dataset_path}/data_aishell.tgz")
        tar = tarfile.open(f"{self.configs.dataset.dataset_path}/data_aishell.tgz", mode="r:gz")
        tar.extractall(self.configs.dataset.dataset_path)
        tar.close()
        os.remove(f"{self.configs.dataset.dataset_path}/data_aishell.tgz")
        self.configs.dataset.dataset_path = os.path.join(self.configs.dataset.dataset_path, "data_aishell")

    def _generate_manifest_files(self, manifest_file_path: str) -> None:
        generate_character_labels(
            dataset_path=self.configs.dataset.dataset_path,
            vocab_path=self.configs.tokenizer.vocab_path,
        )
        generate_character_script(
            dataset_path=self.configs.dataset.dataset_path,
            manifest_file_path=manifest_file_path,
            vocab_path=self.configs.tokenizer.vocab_path,
        )

    def _parse_manifest_file(self, manifest_file_path: str) -> Tuple[list, list]:
        """Parsing manifest file"""
        audio_paths = list()
        transcripts = list()

        with open(manifest_file_path) as f:
            for idx, line in enumerate(f.readlines()):
                audio_path, _, transcript = line.split("\t")
                transcript = transcript.replace("\n", "")

                audio_paths.append(audio_path)
                transcripts.append(transcript)

        return audio_paths, transcripts

    def prepare_data(self) -> None:
        r"""
        Prepare AI-Shell manifest file. If there is not exist manifest file, generate manifest file.

        Returns:
            tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.
        """
        if self.configs.dataset.dataset_download:
            self._download_dataset()

        if not os.path.exists(self.configs.dataset.manifest_file_path):
            self.logger.info("Manifest file is not exists !!\n" "Generate manifest files..")
            if not os.path.exists(self.configs.dataset.dataset_path):
                raise ValueError("Dataset path is not valid.")
            self._generate_manifest_files(self.configs.dataset.manifest_file_path)

    def setup(self, stage: Optional[str] = None) -> None:
        r"""
        Split `train` and `valid` dataset for training.

        Args:
            stage (str): stage of training. `train` or `valid`
            tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

        Returns:
            None
        """
        valid_end_idx = self.AISHELL_TRAIN_NUM + self.AISHELL_VALID_NUM
        audio_paths, transcripts = self._parse_manifest_file(self.configs.dataset.manifest_file_path)

        audio_paths = {
            "train": audio_paths[: self.AISHELL_TRAIN_NUM],
            "valid": audio_paths[self.AISHELL_TRAIN_NUM : valid_end_idx],
            "test": audio_paths[valid_end_idx:],
        }
        transcripts = {
            "train": transcripts[: self.AISHELL_TRAIN_NUM],
            "valid": transcripts[self.AISHELL_TRAIN_NUM : valid_end_idx],
            "test": transcripts[valid_end_idx:],
        }

        for stage in audio_paths.keys():
            self.dataset[stage] = SpeechToTextDataset(
                configs=self.configs,
                dataset_path=self.configs.dataset.dataset_path,
                audio_paths=audio_paths[stage],
                transcripts=transcripts[stage],
                apply_spec_augment=self.configs.audio.apply_spec_augment if stage == "train" else False,
                del_silence=self.configs.audio.del_silence if stage == "train" else False,
            )

    def train_dataloader(self) -> AudioDataLoader:
        sampler = SmartBatchingSampler if self.configs.trainer.sampler == "smart" else RandomSampler
        train_sampler = sampler(data_source=self.dataset["train"], batch_size=self.configs.trainer.batch_size)
        return AudioDataLoader(
            dataset=self.dataset["train"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=train_sampler,
        )

    def val_dataloader(self) -> AudioDataLoader:
        sampler = SmartBatchingSampler if self.configs.trainer.sampler == "smart" else RandomSampler
        valid_sampler = sampler(self.dataset["valid"], batch_size=self.configs.trainer.batch_size)
        return AudioDataLoader(
            dataset=self.dataset["valid"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=valid_sampler,
        )

    def test_dataloader(self) -> AudioDataLoader:
        sampler = SmartBatchingSampler if self.configs.trainer.sampler == "smart" else RandomSampler
        test_sampler = sampler(self.dataset["test"], batch_size=self.configs.trainer.batch_size)
        return AudioDataLoader(
            dataset=self.dataset["test"],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=test_sampler,
        )
