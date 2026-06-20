# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho

"""Tests for the PL 2.x callback-hook migration in openspeech/callbacks.py."""

from openspeech.callbacks import CheckpointEveryNSteps


def test_uses_pl2_hook_name():
    assert hasattr(CheckpointEveryNSteps, "on_train_batch_end")
    # The legacy PL 1.x hook (silently never fired in PL 2.x) must be gone.
    assert "on_batch_end" not in CheckpointEveryNSteps.__dict__


def _make_trainer(tmp_path, global_step):
    class _CkptCallback:
        dirpath = str(tmp_path)
        filename = "model"

    class _Trainer:
        current_epoch = 1
        checkpoint_callback = _CkptCallback()

        def __init__(self, step):
            self.global_step = step
            self.saved = []

        def save_checkpoint(self, path):
            self.saved.append(path)

    return _Trainer(global_step)


def test_saves_on_frequency(tmp_path):
    cb = CheckpointEveryNSteps(save_step_frequency=10)
    trainer = _make_trainer(tmp_path, global_step=20)
    cb.on_train_batch_end(trainer, pl_module=None, outputs=None, batch=None, batch_idx=0)
    assert len(trainer.saved) == 1
    assert trainer.saved[0].endswith("1_20.ckpt")


def test_skips_off_frequency(tmp_path):
    cb = CheckpointEveryNSteps(save_step_frequency=10)
    trainer = _make_trainer(tmp_path, global_step=7)
    cb.on_train_batch_end(trainer, pl_module=None, outputs=None, batch=None, batch_idx=0)
    assert trainer.saved == []
