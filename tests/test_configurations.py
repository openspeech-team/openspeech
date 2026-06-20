# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho

"""Tests for the trainer dataclass migration in openspeech/dataclass/configurations.py."""

from openspeech.dataclass.configurations import (
    BaseTrainerConfigs,
    CPUResumeTrainerConfigs,
    CPUTrainerConfigs,
    Fp16GPUTrainerConfigs,
    GPUResumeTrainerConfigs,
    GPUTrainerConfigs,
    TPUResumeTrainerConfigs,
)


def test_strategy_field_replaces_accelerator():
    cfg = BaseTrainerConfigs()
    assert cfg.strategy == "auto"
    assert not hasattr(cfg, "accelerator")


def test_default_logger_is_tensorboard():
    assert BaseTrainerConfigs().logger == "tensorboard"


def test_removed_legacy_fields_absent():
    for cls in (BaseTrainerConfigs, GPUTrainerConfigs, Fp16GPUTrainerConfigs):
        cfg = cls()
        assert not hasattr(cfg, "auto_scale_batch_size")
        assert not hasattr(cfg, "auto_select_gpus")
        assert not hasattr(cfg, "amp_backend")


def test_resume_configs_have_checkpoint_path():
    for cls in (CPUResumeTrainerConfigs, GPUResumeTrainerConfigs, TPUResumeTrainerConfigs):
        assert "checkpoint_path" in cls.__dataclass_fields__


def test_nonresume_config_lacks_checkpoint_path():
    assert "checkpoint_path" not in CPUTrainerConfigs.__dataclass_fields__
