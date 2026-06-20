# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho

"""Tests for the Hydra/PL 2.x migration of the openspeech_cli entrypoints."""

import importlib

import pytest

CLI_MODULES = [
    "openspeech_cli.hydra_train",
    "openspeech_cli.hydra_lm_train",
    "openspeech_cli.hydra_eval",
    "openspeech_cli.hydra_ensemble_eval",
]


@pytest.mark.parametrize("module_name", CLI_MODULES)
def test_cli_module_imports(module_name):
    importlib.import_module(module_name)


def test_modern_rank_zero_import_path():
    from pytorch_lightning.utilities.rank_zero import rank_zero_info  # noqa: F401


def test_hydra_main_uses_version_base():
    for module_name in CLI_MODULES:
        module = importlib.import_module(module_name)
        with open(module.__file__) as f:
            source = f.read()
        assert "version_base=None" in source


def test_resume_guard_present():
    for module_name in ("openspeech_cli.hydra_train", "openspeech_cli.hydra_lm_train"):
        module = importlib.import_module(module_name)
        with open(module.__file__) as f:
            source = f.read()
        assert "ckpt_path=ckpt_path" in source
        assert "requires `trainer.checkpoint_path`" in source
