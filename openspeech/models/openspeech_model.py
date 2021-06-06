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

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional, List, Tuple
from omegaconf import DictConfig
from torch import Tensor

from openspeech.criterion import CRITERION_REGISTRY
from openspeech.metrics import WordErrorRate, CharacterErrorRate
from openspeech.optim.scheduler import SCHEDULER_REGISTRY
from openspeech.utils import get_class_name
from openspeech.vocabs.vocab import Vocabulary


class OpenspeechModel(pl.LightningModule):
    r"""
    Super class of openspeech models.

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        configs (DictConfig): configuration set.
        vocab (Vocabulary): the class of vocabulary

    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
            `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        * y_hats (torch.FloatTensor): Result of model predictions.
    """
    def __init__(self, configs: DictConfig, vocab: Vocabulary) -> None:
        super(OpenspeechModel, self).__init__()
        self.configs = configs
        self.num_classes = len(vocab)
        self.gradient_clip_val = configs.trainer.gradient_clip_val
        self.vocab = vocab
        self.current_val_loss = 100.0
        self.wer_metric = WordErrorRate(vocab)
        self.cer_metric = CharacterErrorRate(vocab)
        self.criterion = self.configure_criterion(configs.criterion.criterion_name)

    def build_model(self):
        raise NotImplementedError

    def set_beam_decoder(self, beam_size: int = 3):
        raise NotImplementedError

    def log_steps(
            self,
            stage: str,
            wer: float,
            cer: float,
            loss: Optional[float] = None,
            cross_entropy_loss: Optional[float] = None,
            ctc_loss: Optional[float] = None,
    ) -> None:
        r"""
        Provides log dictionary.

        Args:
            stage (str): current stage (train, valid, test)
            wer (float): word error rate
            cer (float): character error rate
            loss (float): loss of model's prediction
            cross_entropy_loss (Optional, float): cross entropy loss of model's prediction
            ctc_loss (Optional, float): ctc loss of model's prediction
        """
        self.log(f"{stage}_wer", wer)
        self.log(f"{stage}_cer", cer)
        if loss is not None:
            self.log(f"{stage}_loss", loss)
        if cross_entropy_loss is not None:
            self.log(f"{stage}_cross_entropy_loss", cross_entropy_loss)
        if ctc_loss is not None:
            self.log(f"{stage}_ctc_loss", ctc_loss)
        if hasattr(self, "optimizer"):
            self.log("current_lr", self.get_lr())

    def forward(self, inputs: torch.FloatTensor, input_lengths: torch.LongTensor) -> Dict[str, Tensor]:
        r"""
        Forward propagate a `inputs` and `targets` pair for inference.

        Inputs:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * outputs (dict): Result of model predictions.
        """
        raise NotImplementedError

    def training_step(self, batch: tuple, batch_idx: int):
        r"""
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        raise NotImplementedError

    def validation_step(self, batch: tuple, batch_idx: int):
        r"""
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        raise NotImplementedError

    def validation_epoch_end(self, outputs: dict) -> dict:
        self.current_val_loss = torch.stack([output['val_loss'] for output in outputs]).mean()

        if get_class_name(self.scheduler) == "WarmupReduceLROnPlateauScheduler" \
            or get_class_name(self.scheduler) == "ReduceLROnPlateauScheduler":
            self.scheduler.step(self.current_val_loss)

        return {
            'loss': self.current_val_loss,
            'log': {'val_loss': self.current_val_loss},
        }

    def test_step(self, batch: tuple, batch_idx: int):
        r"""
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        raise NotImplementedError

    def configure_optimizers(self):
        r"""
        Choose what optimizers and learning-rate schedulers to use in your optimization.


        Returns:
            - **Two lists** - The first list has multiple optimizers, and the second has multiple LR schedulers
                (or multiple ``lr_dict``).
        """
        from torch.optim import Adam, Adagrad, Adadelta, Adamax, AdamW, SGD, ASGD
        from openspeech.optim import AdamP, RAdam, Novograd

        SUPPORTED_OPTIMIZERS = {
            "adam": Adam,
            "adamp": AdamP,
            "radam": RAdam,
            "adagrad": Adagrad,
            "adadelta": Adadelta,
            "adamax": Adamax,
            "adamw": AdamW,
            "sgd": SGD,
            "asgd": ASGD,
            "novograd": Novograd,
        }

        assert self.configs.model.optimizer in SUPPORTED_OPTIMIZERS.keys(), \
            f"Unsupported Optimizer: {self.configs.model.optimizer}\n" \
            f"Supported Optimizers: {SUPPORTED_OPTIMIZERS.keys()}"

        self.optimizer = SUPPORTED_OPTIMIZERS[self.configs.model.optimizer](
            self.parameters(),
            lr=self.configs.lr_scheduler.lr,
        )
        self.scheduler = SCHEDULER_REGISTRY[self.configs.lr_scheduler.scheduler_name](
            optimizer=self.optimizer,
            configs=self.configs,
        )
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': self.scheduler,
                'monitor': 'metric_to_track',
            }
        }

    def configure_criterion(self, criterion_name: str) -> nn.Module:
        r"""
        Configure criterion for training.

        Args:
            criterion_name (str): name of criterion

        Returns:
            criterion (nn.Module): criterion for training
        """
        if criterion_name in ('joint_ctc_cross_entropy', 'label_smoothed_cross_entropy'):
            return CRITERION_REGISTRY[criterion_name](
                configs=self.configs,
                num_classes=self.num_classes,
                vocab=self.vocab,
            )
        else:
            return CRITERION_REGISTRY[criterion_name](
                configs=self.configs,
                vocab=self.vocab,
            )

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
