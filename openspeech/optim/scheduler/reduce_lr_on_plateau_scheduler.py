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

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import DictConfig
from torch.optim import Optimizer

from openspeech.dataclass.configurations import LearningRateSchedulerConfigs
from openspeech.optim.scheduler import register_scheduler
from openspeech.optim.scheduler.lr_scheduler import LearningRateScheduler


@dataclass
class ReduceLROnPlateauConfigs(LearningRateSchedulerConfigs):
    scheduler_name: str = field(default="reduce_lr_on_plateau", metadata={"help": "Name of learning rate scheduler."})
    lr_patience: int = field(
        default=1, metadata={"help": "Number of epochs with no improvement after which learning rate will be reduced."}
    )
    lr_factor: float = field(
        default=0.3, metadata={"help": "Factor by which the learning rate will be reduced. new_lr = lr * factor."}
    )


@register_scheduler("reduce_lr_on_plateau", dataclass=ReduceLROnPlateauConfigs)
class ReduceLROnPlateauScheduler(LearningRateScheduler):
    r"""
    Reduce learning rate when a metric has stopped improving. Models often benefit from reducing the learning rate by
    a factor of 2-10 once learning stagnates. This scheduler reads a metrics quantity and if no improvement is seen
    for a ‘patience’ number of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): wrapped optimizer.
        configs (DictConfig): configuration set.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        configs: DictConfig,
    ) -> None:
        super(ReduceLROnPlateauScheduler, self).__init__(optimizer, configs.lr_scheduler.lr)
        self.lr = configs.lr_scheduler.lr
        self.lr_patience = configs.lr_scheduler.lr_patience
        self.lr_factor = configs.lr_scheduler.lr_factor
        self.val_loss = 100.0
        self.count = 0

    def step(self, val_loss: Optional[float] = None):
        if val_loss is not None:
            if self.val_loss < val_loss:
                self.count += 1
                self.val_loss = val_loss
            else:
                self.count = 0
                self.val_loss = val_loss

            if self.lr_patience == self.count:
                self.count = 0
                self.lr *= self.lr_factor
                self.set_lr(self.optimizer, self.lr)

        return self.lr
