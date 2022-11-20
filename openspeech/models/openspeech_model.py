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

from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import ASGD, SGD, Adadelta, Adagrad, Adam, Adamax, AdamW

from openspeech.criterion import CRITERION_REGISTRY
from openspeech.metrics import CharacterErrorRate, WordErrorRate
from openspeech.optim import AdamP, Novograd, RAdam
from openspeech.optim.scheduler import SCHEDULER_REGISTRY
from openspeech.tokenizers.tokenizer import Tokenizer


class OpenspeechModel(pl.LightningModule):
    r"""
    Super class of openspeech models.

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions.
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(OpenspeechModel, self).__init__()
        self.configs = configs
        self.num_classes = len(tokenizer)
        self.tokenizer = tokenizer
        self.current_val_loss = 100.0
        self.wer_metric = WordErrorRate(tokenizer)
        self.cer_metric = CharacterErrorRate(tokenizer)
        if hasattr(configs, "trainer"):
            self.gradient_clip_val = configs.trainer.gradient_clip_val
        if hasattr(configs, "criterion"):
            self.criterion = self.configure_criterion(configs.criterion.criterion_name)

    def set_beam_decoder(self, beam_size: int = 3):
        raise NotImplementedError

    def info(self, dictionary: dict) -> None:
        r"""
        Logging information from dictionary.

        Args:
            dictionary (dict): dictionary contains information.
        """
        for key, value in dictionary.items():
            self.log(key, value, prog_bar=True)

    def forward(self, inputs: torch.FloatTensor, input_lengths: torch.LongTensor) -> Dict[str, Tensor]:
        r"""
        Forward propagate a `inputs` and `targets` pair for inference.

        Inputs:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            outputs (dict): Result of model predictions.
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
            - **Dictionary** - The first item has multiple optimizers, and the second has multiple LR schedulers (or multiple ``lr_dict``).
        """
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

        assert self.configs.model.optimizer in SUPPORTED_OPTIMIZERS.keys(), (
            f"Unsupported Optimizer: {self.configs.model.optimizer}\n"
            f"Supported Optimizers: {SUPPORTED_OPTIMIZERS.keys()}"
        )

        self.optimizer = SUPPORTED_OPTIMIZERS[self.configs.model.optimizer](
            self.parameters(),
            lr=self.configs.lr_scheduler.lr,
        )
        scheduler = SCHEDULER_REGISTRY[self.configs.lr_scheduler.scheduler_name](self.optimizer, self.configs)

        if self.configs.lr_scheduler.scheduler_name == "reduce_lr_on_plateau":
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            }
        elif self.configs.lr_scheduler.scheduler_name == "warmup_reduce_lr_on_plateau":
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "step",
            }
        else:
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
            }

        return [self.optimizer], [lr_scheduler]

    def configure_criterion(self, criterion_name: str) -> nn.Module:
        r"""
        Configure criterion for training.

        Args:
            criterion_name (str): name of criterion

        Returns:
            criterion (nn.Module): criterion for training
        """
        if criterion_name in ("joint_ctc_cross_entropy", "label_smoothed_cross_entropy"):
            return CRITERION_REGISTRY[criterion_name](
                configs=self.configs,
                num_classes=self.num_classes,
                tokenizer=self.tokenizer,
            )
        else:
            return CRITERION_REGISTRY[criterion_name](
                configs=self.configs,
                tokenizer=self.tokenizer,
            )

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g["lr"]

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr
