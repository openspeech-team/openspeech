# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho

"""Tests for OpenspeechTask — the Lightning wrapper added in Phase 2 (Approach A).

Covers step delegation, metric logging (sync_dist), the `model.` state_dict prefix, and the
legacy-checkpoint key remap that keeps pre-Phase-2 `.ckpt` files loadable.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn

from openspeech.task import OpenspeechTask


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        return {"loss": torch.tensor(1.0), "logs": {"train_loss": 0.5}}

    def validation_step(self, batch, batch_idx):
        return {"loss": torch.tensor(2.0), "logs": {"val_loss": 0.25}}

    def test_step(self, batch, batch_idx):
        return {"loss": torch.tensor(3.0), "logs": {"test_loss": 0.1}}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_task_is_lightning_module():
    assert issubclass(OpenspeechTask, pl.LightningModule)


def test_task_delegates_training_step():
    task = OpenspeechTask(_FakeModel())
    assert task.training_step(("batch",), 0)["logs"] == {"train_loss": 0.5}


def test_task_delegates_configure_optimizers():
    task = OpenspeechTask(_FakeModel())
    assert isinstance(task.configure_optimizers(), torch.optim.SGD)


def test_task_state_dict_is_model_prefixed():
    keys = list(OpenspeechTask(_FakeModel()).state_dict().keys())
    assert keys and all(key.startswith("model.") for key in keys)


def test_task_logs_step_outputs_with_sync_dist():
    task = OpenspeechTask(_FakeModel())
    calls = []
    task.log = lambda key, value, **kwargs: calls.append((key, kwargs))
    task.on_validation_batch_end({"loss": 1.0, "logs": {"val_loss": 0.25}}, batch=None, batch_idx=0)
    assert calls == [("val_loss", {"prog_bar": True, "sync_dist": True})]


def test_task_logs_noop_without_logs_key():
    task = OpenspeechTask(_FakeModel())
    calls = []
    task.log = lambda *args, **kwargs: calls.append(args)
    task.on_test_batch_end({"loss": 1.0}, batch=None, batch_idx=0)
    assert calls == []


def test_on_load_checkpoint_remaps_legacy_keys():
    task = OpenspeechTask(_FakeModel())
    legacy = {"state_dict": {"linear.weight": torch.zeros(2, 2), "linear.bias": torch.zeros(2)}}
    task.on_load_checkpoint(legacy)
    assert set(legacy["state_dict"].keys()) == {"model.linear.weight", "model.linear.bias"}


def test_on_load_checkpoint_leaves_prefixed_keys_untouched():
    task = OpenspeechTask(_FakeModel())
    already = {"state_dict": {"model.linear.weight": torch.zeros(2, 2)}}
    task.on_load_checkpoint(already)
    assert set(already["state_dict"].keys()) == {"model.linear.weight"}


def test_legacy_state_dict_loads_into_task_after_remap():
    legacy_state = dict(_FakeModel().state_dict())  # un-prefixed (pre-Phase-2 format)
    checkpoint = {"state_dict": legacy_state}
    task = OpenspeechTask(_FakeModel())
    task.on_load_checkpoint(checkpoint)
    missing, unexpected = task.load_state_dict(checkpoint["state_dict"], strict=False)
    assert not unexpected
