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

import math
import torch
from typing import Optional
from dataclasses import dataclass, field
from omegaconf import DictConfig
from torch.optim import Optimizer

from openspeech.dataclass.configurations import LearningRateSchedulerConfigs
from openspeech.optim.scheduler import register_scheduler
from openspeech.optim.scheduler.lr_scheduler import LearningRateScheduler


@dataclass
class TransformerLRSchedulerConfigs(LearningRateSchedulerConfigs):
    scheduler_name: str = field(
        default="transformer", metadata={"help": "Name of learning rate scheduler."}
    )
    peak_lr: float = field(
        default=1e-04, metadata={"help": "Maximum learning rate."}
    )
    final_lr: float = field(
        default=1e-07, metadata={"help": "Final learning rate."}
    )
    final_lr_scale: float = field(
        default=0.05, metadata={"help": "Final learning rate scale"}
    )
    warmup_steps: int = field(
        default=10000, metadata={"help": "Warmup the learning rate linearly for the first N updates"}
    )
    decay_steps: int = field(
        default=150000, metadata={"help": "Steps in decay stages"}
    )


@register_scheduler("transformer", dataclass=TransformerLRSchedulerConfigs)
class TransformerLRScheduler(LearningRateScheduler):
    r"""
    Transformer Learning Rate Scheduler proposed in "Attention Is All You Need"

    Args:
        optimizer (Optimizer): wrapped optimizer.
        configs (DictConfig): configuration set.
    """
    def __init__(
            self,
            optimizer: Optimizer,
            configs: DictConfig,
    ) -> None:
        assert isinstance(configs.lr_scheduler.warmup_steps, int), "warmup_steps should be inteager type"
        assert isinstance(configs.lr_scheduler.decay_steps, int), "total_steps should be inteager type"

        super(TransformerLRScheduler, self).__init__(optimizer, 0.0)
        self.final_lr = configs.lr_scheduler.final_lr
        self.peak_lr = configs.lr_scheduler.peak_lr
        self.warmup_steps = configs.lr_scheduler.warmup_steps
        self.decay_steps = configs.lr_scheduler.decay_steps

        self.warmup_rate = self.peak_lr / self.warmup_steps
        self.decay_factor = -math.log(configs.lr_scheduler.final_lr_scale) / self.decay_steps

        self.lr = self.init_lr
        self.update_step = 0

    def _decide_stage(self):
        if self.update_step < self.warmup_steps:
            return 0, self.update_step

        if self.warmup_steps <= self.update_step < self.warmup_steps + self.decay_steps:
            return 1, self.update_step - self.warmup_steps

        return 2, None

    def step(self, val_loss: Optional[torch.FloatTensor] = None):
        self.update_step += 1
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.update_step * self.warmup_rate
        elif stage == 1:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 2:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)

        return self.lr
