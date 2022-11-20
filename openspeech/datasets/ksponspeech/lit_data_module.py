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
from typing import Optional

import pytorch_lightning as pl
from omegaconf import DictConfig

from openspeech.data.audio.data_loader import AudioDataLoader
from openspeech.data.audio.dataset import SpeechToTextDataset
from openspeech.data.sampler import RandomSampler, SmartBatchingSampler
from openspeech.datasets import register_data_module
from openspeech.datasets.ksponspeech.preprocess.character import generate_character_labels, generate_character_script
from openspeech.datasets.ksponspeech.preprocess.grapheme import sentence_to_grapheme
from openspeech.datasets.ksponspeech.preprocess.preprocess import preprocess, preprocess_test_data
from openspeech.datasets.ksponspeech.preprocess.subword import sentence_to_subwords, train_sentencepiece
from openspeech.tokenizers.tokenizer import Tokenizer


@register_data_module("ksponspeech")
class LightningKsponSpeechDataModule(pl.LightningDataModule):
    r"""
    Lightning data module for KsponSpeech. KsponSpeech corpus contains 969 h of general open-domain dialog utterances,
    spoken by about 2000 native Korean speakers in a clean environment. All data were constructed by recording the
    dialogue of two people freely conversing on a variety of topics and manually transcribing the utterances.
    The transcription provides a dual transcription consisting of orthography and pronunciation,
    and disfluency tags for spontaneity of speech, such as filler words, repeated words, and word fragments.

    Attributes:
        KSPONSPEECH_TRAIN_NUM (int): the number of KsponSpeech's train data.
        KSPONSPEECH_VALID_NUM (int): the number of KsponSpeech's validation data.
        KSPONSPEECH_TEST_NUM (int): the number of KsponSpeech's test data.

    Args:
        configs (DictConfig): configuration set.
    """
    KSPONSPEECH_TRAIN_NUM = 620000
    KSPONSPEECH_VALID_NUM = 2545
    KSPONSPEECH_TEST_NUM = 6000

    def __init__(self, configs: DictConfig) -> None:
        super(LightningKsponSpeechDataModule, self).__init__()
        self.configs = configs
        self.dataset = dict()
        self.logger = logging.getLogger(__name__)
        self.encoding = "cp949" if self.configs.tokenizer.unit == "kspon_grapheme" else "utf-8"

    def _generate_manifest_files(self, manifest_file_path: str) -> None:
        r"""
        Generate KsponSpeech manifest file.
        Format: AUDIO_PATH [TAB] TEXT_TRANSCRIPTS [TAB] NUMBER_TRANSCRIPT
        """
        train_valid_audio_paths, train_valid_transcripts = preprocess(
            self.configs.dataset.dataset_path, self.configs.dataset.preprocess_mode
        )
        test_audio_paths, test_transcripts = preprocess_test_data(
            self.configs.dataset.test_manifest_dir, self.configs.dataset.preprocess_mode
        )

        audio_paths = train_valid_audio_paths + test_audio_paths
        transcripts = train_valid_transcripts + test_transcripts

        if self.configs.tokenizer.unit == "kspon_character":
            generate_character_labels(transcripts, self.configs.tokenizer.vocab_path)
            generate_character_script(audio_paths, transcripts, manifest_file_path, self.configs.tokenizer.vocab_path)

        elif self.configs.tokenizer.unit == "kspon_subword":
            train_sentencepiece(transcripts, self.configs.tokenizer.vocab_size, self.configs.tokenizer.blank_token)
            sentence_to_subwords(
                audio_paths, transcripts, manifest_file_path, sp_model_path=self.configs.tokenizer.sp_model_path
            )

        elif self.configs.tokenizer.unit == "kspon_grapheme":
            sentence_to_grapheme(audio_paths, transcripts, manifest_file_path, self.configs.tokenizer.vocab_path)

        else:
            raise ValueError(f"Unsupported vocab : {self.configs.tokenizer.unit}")

    def _parse_manifest_file(self):
        r"""
        Parsing manifest file.

        Returns:
            audio_paths (list): list of audio path
            transcritps (list): list of transcript of audio
        """
        audio_paths = list()
        transcripts = list()

        with open(self.configs.dataset.manifest_file_path, encoding=self.encoding) as f:
            for idx, line in enumerate(f.readlines()):
                audio_path, korean_transcript, transcript = line.split("\t")
                transcript = transcript.replace("\n", "")

                audio_paths.append(audio_path)
                transcripts.append(transcript)

        return audio_paths, transcripts

    def prepare_data(self):
        r"""
        Prepare KsponSpeech manifest file. If there is not exist manifest file, generate manifest file.

        Returns:
            tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.
        """
        if not os.path.exists(self.configs.dataset.manifest_file_path):
            self.logger.info("Manifest file is not exists !!\n" "Generate manifest files..")
            if not os.path.exists(self.configs.dataset.dataset_path):
                raise FileNotFoundError
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
        valid_end_idx = self.KSPONSPEECH_TRAIN_NUM + self.KSPONSPEECH_VALID_NUM

        audio_paths, transcripts = self._parse_manifest_file()
        audio_paths = {
            "train": audio_paths[: self.KSPONSPEECH_TRAIN_NUM],
            "valid": audio_paths[self.KSPONSPEECH_TRAIN_NUM : valid_end_idx],
            "test": audio_paths[valid_end_idx:],
        }
        transcripts = {
            "train": transcripts[: self.KSPONSPEECH_TRAIN_NUM],
            "valid": transcripts[self.KSPONSPEECH_TRAIN_NUM : valid_end_idx],
            "test": transcripts[valid_end_idx:],
        }

        for stage in audio_paths.keys():
            if stage == "test":
                dataset_path = self.configs.dataset.test_dataset_path
            else:
                dataset_path = self.configs.dataset.dataset_path

            self.dataset[stage] = SpeechToTextDataset(
                configs=self.configs,
                dataset_path=dataset_path,
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
