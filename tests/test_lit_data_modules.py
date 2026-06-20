# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho

"""Tests for the data-module migration (self.logger -> self._logger, PL 2.x registration)."""

import inspect

# Importing the modules forces their @register_data_module decorators to run.
import openspeech.datasets.aishell.lit_data_module  # noqa: F401
import openspeech.datasets.ksponspeech.lit_data_module  # noqa: F401
import openspeech.datasets.language_model.lit_data_module  # noqa: F401
import openspeech.datasets.librispeech.lit_data_module  # noqa: F401
import pytorch_lightning as pl
from openspeech.datasets import DATA_MODULE_REGISTRY
from openspeech.datasets.aishell.lit_data_module import LightningAIShellDataModule
from openspeech.datasets.ksponspeech.lit_data_module import LightningKsponSpeechDataModule
from openspeech.datasets.language_model.lit_data_module import LightningLanguageModelDataModule
from openspeech.datasets.librispeech.lit_data_module import LightningLibriSpeechDataModule

DATA_MODULES = [
    LightningKsponSpeechDataModule,
    LightningLibriSpeechDataModule,
    LightningAIShellDataModule,
    LightningLanguageModelDataModule,
]


def test_data_modules_registered():
    for key in ("ksponspeech", "librispeech", "aishell", "lm"):
        assert key in DATA_MODULE_REGISTRY


def test_data_modules_are_lightning_data_modules():
    for cls in DATA_MODULES:
        assert issubclass(cls, pl.LightningDataModule)


def test_logger_attribute_renamed():
    for cls in DATA_MODULES:
        source = inspect.getsource(cls)
        assert "self._logger" in source
        # the bare `self.logger` would shadow nothing in PL 2.x but is intentionally avoided
        assert "self.logger" not in source
