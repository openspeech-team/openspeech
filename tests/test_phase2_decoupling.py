# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho

"""Phase 2 (Lightning decoupling) tests.

collect_outputs is now pure compute (no self.log side effect); logging is centralized in the
LightningModule batch-end hooks reading the "logs" key. These tests pin that contract and the
stage="valid" -> "val" fix without requiring a real Trainer/dataset.
"""

import importlib
import inspect

import pytest

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


def test_model_defines_batch_end_hooks():
    for hook in ("on_train_batch_end", "on_validation_batch_end", "on_test_batch_end", "_log_step_outputs"):
        assert hook in OpenspeechModel.__dict__


def test_log_step_outputs_logs_only_logs_key():
    model = OpenspeechModel.__new__(OpenspeechModel)
    captured = {}
    model.info = lambda d: captured.update(d)
    model.on_train_batch_end({"loss": 1.0, "logs": {"train_loss": 0.5, "learning_rate": 1e-4}}, batch=None, batch_idx=0)
    assert captured == {"train_loss": 0.5, "learning_rate": 1e-4}


def test_log_step_outputs_without_logs_key_is_noop():
    model = OpenspeechModel.__new__(OpenspeechModel)
    calls = []
    model.info = lambda d: calls.append(d)
    model.on_test_batch_end({"loss": 1.0}, batch=None, batch_idx=0)
    assert calls == []


@pytest.mark.parametrize("model_pkg", ["conformer", "squeezeformer", "contextnet", "transformer"])
def test_validation_stage_renamed_to_val(model_pkg):
    module = importlib.import_module(f"openspeech.models.{model_pkg}.model")
    with open(module.__file__) as f:
        source = f.read()
    assert 'stage="valid"' not in source
