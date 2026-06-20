# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho

"""Tests for the PyTorch-Lightning 2.x trainer-building logic in openspeech/utils.py."""

import pytorch_lightning as pl
import pytest
from omegaconf import OmegaConf

from openspeech.dataclass import configurations as C
from openspeech.utils import get_pl_trainer, resolve_accelerator_settings

# name -> (accelerator, devices, precision) with num_devices=0, tpu_cores=8
EXPECTED = {
    "cpu": ("cpu", "auto", "32-true"),
    "gpu": ("gpu", "auto", "32-true"),
    "tpu": ("tpu", 8, "32-true"),
    "gpu-fp16": ("gpu", "auto", "16-mixed"),
    "tpu-fp16": ("tpu", 8, "16-mixed"),
    "cpu-fp64": ("cpu", "auto", "64-true"),
    "cpu-resume": ("cpu", "auto", "32-true"),
    "gpu-resume": ("gpu", "auto", "32-true"),
    "tpu-resume": ("tpu", 8, "32-true"),
}


@pytest.mark.parametrize("name,expected", list(EXPECTED.items()))
def test_resolve_accelerator_settings(name, expected):
    settings = resolve_accelerator_settings(name, num_devices=0, tpu_cores=8)
    assert (settings["accelerator"], settings["devices"], settings["precision"]) == expected


def test_resolve_gpu_uses_num_devices():
    assert resolve_accelerator_settings("gpu", num_devices=4)["devices"] == 4


def test_resolve_unsupported_name_raises():
    with pytest.raises(ValueError):
        resolve_accelerator_settings("quantum", num_devices=0)


# TPU trainers are excluded here: constructing pl.Trainer(accelerator="tpu") requires
# torch_xla. The name->settings mapping for tpu is covered by test_resolve_accelerator_settings.
_CONFIG_CLASSES = {
    "cpu": C.CPUTrainerConfigs,
    "gpu": C.GPUTrainerConfigs,
    "gpu-fp16": C.Fp16GPUTrainerConfigs,
    "cpu-fp64": C.Fp64CPUTrainerConfigs,
}


@pytest.mark.parametrize("name", list(_CONFIG_CLASSES.keys()))
def test_get_pl_trainer_builds(name):
    configs = OmegaConf.create({"trainer": OmegaConf.structured(_CONFIG_CLASSES[name]())})
    trainer = get_pl_trainer(configs, num_devices=0, logger=True)
    assert isinstance(trainer, pl.Trainer)
