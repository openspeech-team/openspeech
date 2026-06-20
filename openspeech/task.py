# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytorch_lightning as pl


class OpenspeechTask(pl.LightningModule):
    r"""
    PyTorch-Lightning task that wraps a framework-agnostic ``OpenspeechModel`` (an ``nn.Module``).

    Phase 2 decoupling separates concerns:
        - the **model** owns pure computation: ``forward``, the per-paradigm step methods,
          ``collect_outputs``, ``configure_optimizers``, metrics and criterion.
        - the **task** owns the Lightning surface: step-hook delegation, optimizer wiring and
          the metric logging that each step's ``collect_outputs`` placed under the ``logs`` key.

    Checkpoint compatibility:
        Wrapping the model as ``self.model`` prefixes every ``state_dict`` key with ``model.``.
        Pre-Phase-2 checkpoints have un-prefixed keys (``encoder.*``, ``fc.*``, ``criterion.*``);
        :meth:`on_load_checkpoint` remaps those onto ``model.*`` so old ``.ckpt`` files still load.

    Args:
        model (OpenspeechModel): the framework-agnostic model to train/evaluate.
    """

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.model.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.model.test_step(batch, batch_idx)

    def configure_optimizers(self):
        return self.model.configure_optimizers()

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        self._log_step_outputs(outputs)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        self._log_step_outputs(outputs)

    def on_test_batch_end(self, outputs, batch, batch_idx, dataloader_idx: int = 0) -> None:
        self._log_step_outputs(outputs)

    def _log_step_outputs(self, outputs) -> None:
        r"""Log the metrics a step's ``collect_outputs`` placed under the ``logs`` key."""
        if isinstance(outputs, dict) and "logs" in outputs:
            for key, value in outputs["logs"].items():
                self.log(key, value, prog_bar=True, sync_dist=True)

    def on_load_checkpoint(self, checkpoint) -> None:
        r"""Remap legacy (pre-Phase-2) un-prefixed ``state_dict`` keys onto ``model.*``."""
        state_dict = checkpoint.get("state_dict")
        if state_dict and not any(key.startswith("model.") for key in state_dict):
            checkpoint["state_dict"] = {f"model.{key}": value for key, value in state_dict.items()}
