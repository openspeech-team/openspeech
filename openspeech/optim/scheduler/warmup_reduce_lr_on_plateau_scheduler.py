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

from omegaconf import DictConfig
from torch.optim import Optimizer
from dataclasses import dataclass, field
from typing import Optional

from openspeech.dataclass.configurations import LearningRateSchedulerConfigs
from openspeech.optim.scheduler import register_scheduler
from openspeech.optim.scheduler.lr_scheduler import LearningRateScheduler
from openspeech.optim.scheduler.reduce_lr_on_plateau_scheduler import ReduceLROnPlateauScheduler
from openspeech.optim.scheduler.warmup_scheduler import WarmupLRScheduler


@dataclass
class WarmupReduceLROnPlateauConfigs(LearningRateSchedulerConfigs):
    scheduler_name: str = field(
        default="warmup_reduce_lr_on_plateau", metadata={"help": "Name of learning rate scheduler."}
    )
    lr_patience: int = field(
        default=1, metadata={"help": "Number of epochs with no improvement after which learning rate will be reduced."}
    )
    lr_factor: float = field(
        default=0.3, metadata={"help": "Factor by which the learning rate will be reduced. new_lr = lr * factor."}
    )
    peak_lr: float = field(
        default=1e-04, metadata={"help": "Maximum learning rate."}
    )
    init_lr: float = field(
        default=1e-10, metadata={"help": "Initial learning rate."}
    )
    warmup_steps: int = field(
        default=4000, metadata={"help": "Warmup the learning rate linearly for the first N updates"}
    )


@register_scheduler("warmup_reduce_lr_on_plateau", dataclass=WarmupReduceLROnPlateauConfigs)
class WarmupReduceLROnPlateauScheduler(LearningRateScheduler):
    r"""
    Warmup learning rate until `warmup_steps` and reduce learning rate on plateau after.

    Args:
        optimizer (Optimizer): wrapped optimizer.
        configs (DictConfig): configuration set.
    """
    def __init__(
            self,
            optimizer: Optimizer,
            configs: DictConfig,
    ) -> None:
        super(WarmupReduceLROnPlateauScheduler, self).__init__(optimizer, configs.lr_scheduler.lr)
        self.warmup_steps = configs.lr_scheduler.warmup_steps
        self.update_steps = 0
        self.warmup_rate = (configs.lr_scheduler.peak_lr - configs.lr_scheduler.init_lr) / self.warmup_steps \
            if self.warmup_steps != 0 else 0
        self.schedulers = [
            WarmupLRScheduler(
                optimizer,
                configs,
            ),
            ReduceLROnPlateauScheduler(
                optimizer,
                configs,
            ),
        ]

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps
        else:
            return 1, None

    def step(self, val_loss: Optional[float] = None):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.schedulers[0].step()
        elif stage == 1:
            self.schedulers[1].step(val_loss)

        self.update_steps += 1

        return self.get_lr()
