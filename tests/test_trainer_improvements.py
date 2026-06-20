# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho

"""Tests for the PL 2.x real-GPU-run improvements (sync_dist, devices override, dead-field removal)."""

import pytorch_lightning as pl
import pytest
from omegaconf import OmegaConf

from openspeech.dataclass.configurations import (
    BaseTrainerConfigs,
    Fp16GPUTrainerConfigs,
    Fp16TPUTrainerConfigs,
    Fp64CPUTrainerConfigs,
    GPUTrainerConfigs,
)
from openspeech.models.openspeech_model import OpenspeechModel
from openspeech.utils import get_pl_trainer


def test_info_logs_with_sync_dist():
    model = OpenspeechModel.__new__(OpenspeechModel)
    calls = []
    model.log = lambda key, value, **kwargs: calls.append((key, value, kwargs))
    model.info({"train_loss": 1.0})
    assert calls == [("train_loss", 1.0, {"prog_bar": True, "sync_dist": True})]


def test_devices_field_default_auto():
    assert BaseTrainerConfigs().devices == "auto"


@pytest.mark.parametrize("cls", [Fp16GPUTrainerConfigs, Fp16TPUTrainerConfigs, Fp64CPUTrainerConfigs])
def test_dead_precision_fields_removed(cls):
    assert "precision" not in cls.__dataclass_fields__


def test_devices_override_builds_trainer():
    trainer_cfg = OmegaConf.structured(GPUTrainerConfigs())
    trainer_cfg.devices = "1"
    configs = OmegaConf.create({"trainer": trainer_cfg})
    # num_devices=8 would normally request all 8 GPUs; the override caps it to 1.
    trainer = get_pl_trainer(configs, num_devices=8, logger=True)
    assert isinstance(trainer, pl.Trainer)


def test_ensemble_eval_passes_checkpoint_kwargs():
    import openspeech_cli.hydra_ensemble_eval as module

    with open(module.__file__) as f:
        source = f.read()
    assert "configs=configs, tokenizer=tokenizer" in source
