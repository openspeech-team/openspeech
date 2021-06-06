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
import wget
import tarfile
import logging
import shutil
import pytorch_lightning as pl
from typing import Tuple, Optional
from omegaconf import DictConfig
from openspeech.data.dataset import SpeechToTextDataset
from torch.utils.data import DataLoader

from openspeech.datasets import register_data_module
from openspeech.vocabs import VOCAB_REGISTRY
from openspeech.vocabs.vocab import Vocabulary
from openspeech.data.data_loader import BucketingSampler, AudioDataLoader


@register_data_module('librispeech')
class LightningLibriSpeechDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning Data Module for LibriSpeech Dataset.

    Args:
        configs (DictConfig): configuraion set
    """
    LIBRISPEECH_TRAIN_NUM = 281241
    LIBRISPEECH_VALID_NUM = 5567
    LIBRISPEECH_TEST_NUM = 5559
    LIBRISPEECH_PARTS = [
        'dev-clean',
        'test-clean',
        'dev-other',
        'test-other',
        'train-clean-100',
        'train-clean-360',
        'train-other-500',
    ]

    def __init__(self, configs: DictConfig) -> None:
        super(LightningLibriSpeechDataModule, self).__init__()
        self.configs = configs
        self.dataset = dict()
        self.logger = logging.getLogger(__name__)

    def _parse_manifest_file(self, manifest_file_path: str) -> Tuple[list, list]:
        """ Parsing manifest file """
        audio_paths = list()
        transcripts = list()

        with open(manifest_file_path) as f:
            for idx, line in enumerate(f.readlines()):
                audio_path, _, transcript = line.split('\t')
                transcript = transcript.replace('\n', '')

                audio_paths.append(audio_path)
                transcripts.append(transcript)

        return audio_paths, transcripts

    def _download_dataset(self) -> None:
        """
        Download librispeech dataset.
            - train-960(train-clean-100, train-clean-360, train-other-500)
            - dev-clean
            - dev-other
            - test-clean
            - test-other
        """
        base_url = "http://www.openslr.org/resources/12"
        train_dir = "train-960"

        if not os.path.exists(self.configs.dataset.dataset_path):
            os.mkdir(self.configs.dataset.dataset_path)

        for part in self.LIBRISPEECH_PARTS:
            self.logger.info(f"Librispeech-{part} download..")
            url = f"{base_url}/{part}.tar.gz"
            wget.download(url)
            shutil.move(f"{part}.tar.gz", os.path.join(self.configs.dataset.dataset_path, f"{part}.tar.gz"))

            self.logger.info(f"Un-tarring archive {self.configs.dataset.dataset_path}/{part}.tar.gz")
            tar = tarfile.open(f"{self.configs.dataset.dataset_path}/{part}.tar.gz", mode="r:gz")
            tar.extractall(self.configs.dataset.dataset_path)
            tar.close()
            os.remove(f"{self.configs.dataset.dataset_path}/{part}.tar.gz")

        self.logger.info("Merge all train packs into one")

        if not os.path.exists(self.configs.dataset.dataset_path):
            os.mkdir(self.configs.dataset.dataset_path)
        if not os.path.exists(os.path.join(self.configs.dataset.dataset_path, train_dir)):
            os.mkdir(os.path.join(self.configs.dataset.dataset_path, train_dir))

        for part in self.LIBRISPEECH_PARTS[-3:]:    # train
            path = os.path.join(self.configs.dataset.dataset_path, part)
            subfolders = os.listdir(path)
            for subfolder in subfolders:
                shutil.move(
                    os.path.join(path, subfolder),
                    os.path.join(self.configs.dataset.dataset_path, train_dir, subfolder),
                )

    def prepare_data(self) -> Vocabulary:
        """
        Prepare librispeech data

        Returns:
            vocab (Vocabulary): vocab class of KsponSpeech.
        """
        if not os.path.exists(self.configs.dataset.dataset_path):
            raise ValueError("Dataset path is not valid.")

        if self.configs.vocab.unit == 'libri_subword':
            from openspeech.datasets.librispeech.preprocess.subword import generate_manifest_files
        elif self.configs.vocab.unit == 'libri_character':
            from openspeech.datasets.librispeech.preprocess.character import generate_manifest_files
        else:
            raise ValueError(f"Unsupported vocabulary unit: {self.configs.vocab.unit}")

        if self.configs.dataset.dataset_download:
            self._download_dataset()

        if not os.path.exists(self.configs.dataset.manifest_file_path):
            self.logger.info("Manifest file is not exists !!\n"
                             "Generate manifest files..")

            if hasattr(self.configs.vocab, "vocab_size"):
                generate_manifest_files(
                    dataset_path=self.configs.dataset.dataset_path,
                    manifest_file_path=self.configs.dataset.manifest_file_path,
                    vocab_path=self.configs.vocab.vocab_path,
                    vocab_size=self.configs.vocab.vocab_size,
                )
            else:
                generate_manifest_files(
                    dataset_path=self.configs.dataset.dataset_path,
                    manifest_file_path=self.configs.dataset.manifest_file_path,
                    vocab_path=self.configs.vocab.vocab_path,
                )

        return VOCAB_REGISTRY[self.configs.vocab.unit](self.configs)

    def setup(self, stage: Optional[str] = None, vocab: Vocabulary = None) -> None:
        r""" Split dataset into train, valid, and test. """
        valid_end_idx = self.LIBRISPEECH_TRAIN_NUM + self.LIBRISPEECH_VALID_NUM
        audio_paths, transcripts = self._parse_manifest_file(self.configs.dataset.manifest_file_path)

        audio_paths = {
            "train": audio_paths[:self.LIBRISPEECH_TRAIN_NUM],
            "valid": audio_paths[self.LIBRISPEECH_TRAIN_NUM:valid_end_idx],
            "test": audio_paths[valid_end_idx:],
        }
        transcripts = {
            "train": transcripts[:self.LIBRISPEECH_TRAIN_NUM],
            "valid": transcripts[self.LIBRISPEECH_TRAIN_NUM:valid_end_idx],
            "test": transcripts[valid_end_idx:],
        }

        for stage in audio_paths.keys():
            self.dataset[stage] = SpeechToTextDataset(
                configs=self.configs,
                dataset_path=self.configs.dataset.dataset_path,
                audio_paths=audio_paths[stage],
                transcripts=transcripts[stage],
                sos_id=vocab.sos_id,
                eos_id=vocab.eos_id,
                apply_spec_augment=self.configs.audio.apply_spec_augment if stage == 'train' else False,
                del_silence=self.configs.audio.del_silence if stage == 'train' else False,
            )

    def train_dataloader(self) -> DataLoader:
        train_sampler = BucketingSampler(self.dataset['train'], batch_size=self.configs.trainer.batch_size)
        return AudioDataLoader(
            dataset=self.dataset['train'],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=train_sampler,
        )

    def val_dataloader(self) -> DataLoader:
        valid_sampler = BucketingSampler(self.dataset['valid'], batch_size=self.configs.trainer.batch_size)
        return AudioDataLoader(
            dataset=self.dataset['valid'],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=valid_sampler,
        )

    def test_dataloader(self) -> DataLoader:
        test_sampler = BucketingSampler(self.dataset['test'], batch_size=self.configs.trainer.batch_size)
        return AudioDataLoader(
            dataset=self.dataset['test'],
            num_workers=self.configs.trainer.num_workers,
            batch_sampler=test_sampler,
        )
