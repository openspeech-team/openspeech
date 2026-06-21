# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho

"""CPU end-to-end training smoke test.

Exercises the real OpenspeechTask <-> PyTorch-Lightning Trainer.fit integration (step delegation,
configure_optimizers, on_*_batch_end logging) with a tiny dummy model and dummy tensors.
fast_dev_run runs exactly one train + one val batch on CPU, so it is memory-trivial (a few KB)
and safe on low-RAM machines. Model *computation* correctness is covered by the unit tests; this
test only proves the training loop wires up and runs without raising.
"""

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from openspeech.task import OpenspeechTask


class _TinyModel(nn.Module):
    """Minimal stand-in following the OpenspeechModel step contract (returns {'loss', 'logs'})."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
        self.optimizer = None

    def forward(self, inputs):
        return self.fc(inputs)

    def _step(self, batch, stage):
        inputs, targets = batch
        loss = nn.functional.cross_entropy(self.fc(inputs), targets)
        return {"loss": loss, "logs": {f"{stage}_loss": loss.detach()}}

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return self.optimizer


def _dummy_loader():
    inputs = torch.randn(8, 4)
    targets = torch.randint(0, 3, (8,))
    return DataLoader(TensorDataset(inputs, targets), batch_size=4)


def test_cpu_training_smoke_fast_dev_run():
    task = OpenspeechTask(_TinyModel())
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,  # one train + one val batch; no checkpoint/logger overhead
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(task, _dummy_loader(), _dummy_loader())
    assert trainer.state.finished


def test_cpu_test_loop_smoke():
    task = OpenspeechTask(_TinyModel())
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.test(task, _dummy_loader())
    assert trainer.state.finished
