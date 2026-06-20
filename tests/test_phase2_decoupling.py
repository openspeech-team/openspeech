# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho

"""Phase 2 (Lightning decoupling) tests — model side.

collect_outputs is pure compute (no self.log); OpenspeechModel is now a framework-agnostic
nn.Module. The Lightning surface (step-hook dispatch + logging) lives on OpenspeechTask
(covered by test_task.py).
"""

import importlib
import inspect

import pytest
import pytorch_lightning as pl
import torch.nn as nn

from openspeech.models.openspeech_ctc_model import OpenspeechCTCModel
from openspeech.models.openspeech_encoder_decoder_model import OpenspeechEncoderDecoderModel
from openspeech.models.openspeech_language_model import OpenspeechLanguageModel
from openspeech.models.openspeech_model import OpenspeechModel
from openspeech.models.openspeech_transducer_model import OpenspeechTransducerModel

BASE_CLASSES = [
    OpenspeechCTCModel,
    OpenspeechEncoderDecoderModel,
    OpenspeechTransducerModel,
    OpenspeechLanguageModel,
]


@pytest.mark.parametrize("cls", BASE_CLASSES)
def test_collect_outputs_is_log_free(cls):
    source = inspect.getsource(cls.collect_outputs)
    assert "self.info" not in source
    assert "self.log" not in source
    assert '"logs"' in source


def test_openspeech_model_is_plain_nn_module():
    assert issubclass(OpenspeechModel, nn.Module)
    assert not issubclass(OpenspeechModel, pl.LightningModule)
    # logging helpers moved to OpenspeechTask
    assert not hasattr(OpenspeechModel, "info")
    assert not hasattr(OpenspeechModel, "on_train_batch_end")


def test_get_lr_returns_none_without_optimizer():
    model = OpenspeechModel.__new__(OpenspeechModel)
    model.optimizer = None
    assert model.get_lr() is None


@pytest.mark.parametrize("model_pkg", ["conformer", "squeezeformer", "contextnet", "transformer"])
def test_validation_stage_renamed_to_val(model_pkg):
    module = importlib.import_module(f"openspeech.models.{model_pkg}.model")
    with open(module.__file__) as f:
        source = f.read()
    assert 'stage="valid"' not in source
