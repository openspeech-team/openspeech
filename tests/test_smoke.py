# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction.

"""Lightweight, network-free smoke tests that prove the package imports and the
PyTorch-Lightning 2.x trainer builder constructs on CPU. Used as the CI gate."""

import lightning.pytorch as pl
from omegaconf import OmegaConf


def test_import_openspeech():
    import openspeech  # noqa: F401


def test_model_registry_populated():
    from openspeech.models import MODEL_REGISTRY

    assert len(MODEL_REGISTRY) > 0


def test_get_pl_trainer_cpu():
    from openspeech.dataclass.configurations import CPUTrainerConfigs
    from openspeech.utils import get_pl_trainer

    trainer_cfg = OmegaConf.structured(CPUTrainerConfigs())
    configs = OmegaConf.create({"trainer": trainer_cfg})

    trainer = get_pl_trainer(configs, num_devices=0, logger=True)

    assert isinstance(trainer, pl.Trainer)


def test_get_pl_trainer_fp16_precision():
    from openspeech.dataclass.configurations import Fp16GPUTrainerConfigs
    from openspeech.utils import get_pl_trainer

    trainer_cfg = OmegaConf.structured(Fp16GPUTrainerConfigs())
    configs = OmegaConf.create({"trainer": trainer_cfg})

    trainer = get_pl_trainer(configs, num_devices=0, logger=True)

    assert isinstance(trainer, pl.Trainer)
