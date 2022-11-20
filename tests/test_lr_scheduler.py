import unittest

import matplotlib.pyplot as plt
import torch
from torch import optim

from openspeech.optim.optimizer import Optimizer
from openspeech.optim.scheduler.reduce_lr_on_plateau_scheduler import (
    ReduceLROnPlateauConfigs,
    ReduceLROnPlateauScheduler,
)
from openspeech.optim.scheduler.transformer_lr_scheduler import TransformerLRScheduler, TransformerLRSchedulerConfigs
from openspeech.optim.scheduler.tri_stage_lr_scheduler import TriStageLRScheduler, TriStageLRSchedulerConfigs
from openspeech.optim.scheduler.warmup_reduce_lr_on_plateau_scheduler import (
    WarmupReduceLROnPlateauConfigs,
    WarmupReduceLROnPlateauScheduler,
)
from openspeech.optim.scheduler.warmup_scheduler import WarmupLRScheduler, WarmupLRSchedulerConfigs
from openspeech.utils import build_dummy_configs


class TestLRScheduler(unittest.TestCase):
    def test_warmup_lr_scheduler(self):
        configs = build_dummy_configs(scheduler_configs=WarmupLRSchedulerConfigs())

        lr_histories = list()
        total_steps = 15000

        model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]

        optimizer = optim.Adam(model, lr=1e-04)
        scheduler = WarmupLRScheduler(optimizer, configs)

        optimizer = Optimizer(
            optim=optimizer,
            scheduler=scheduler,
            scheduler_period=total_steps,
            max_grad_norm=0.0,
        )

        for timestep in range(total_steps):
            optimizer.step(model)
            lr_histories.append(optimizer.get_lr())

        plt.title("WarmupLRScheduler")
        plt.plot(lr_histories, label="lr", color="#FF6C38", linewidth=2)
        plt.legend()
        plt.grid(True)
        plt.xlabel("timestep", fontsize="large")
        plt.ylabel("lr", fontsize="large")
        plt.savefig("WarmupLRScheduler.png")

    def test_reduce_lr_on_plateau_scheduler(self):
        configs = build_dummy_configs(scheduler_configs=ReduceLROnPlateauConfigs())

        lr_histories = list()
        total_steps = 15000

        model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]

        optimizer = optim.Adam(model, lr=1e-04)
        scheduler = ReduceLROnPlateauScheduler(optimizer, configs)

        optimizer = Optimizer(
            optim=optimizer,
            scheduler=scheduler,
            scheduler_period=total_steps,
            max_grad_norm=0.0,
        )

        for timestep in range(total_steps):
            optimizer.step(model)
            lr_histories.append(optimizer.get_lr())

        plt.title("ReduceLROnPlateauScheduler")
        plt.plot(lr_histories, label="lr", color="#FF6C38", linewidth=2)
        plt.legend()
        plt.grid(True)
        plt.xlabel("timestep", fontsize="large")
        plt.ylabel("lr", fontsize="large")
        plt.savefig("ReduceLROnPlateauScheduler.png")

    def test_transformer_lr_scheduler(self):
        configs = build_dummy_configs(scheduler_configs=TransformerLRSchedulerConfigs())

        lr_histories = list()
        total_steps = 15000

        model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]

        optimizer = optim.Adam(model, lr=1e-04)
        scheduler = TransformerLRScheduler(optimizer, configs)

        optimizer = Optimizer(
            optim=optimizer,
            scheduler=scheduler,
            scheduler_period=total_steps,
            max_grad_norm=0.0,
        )

        for timestep in range(total_steps):
            optimizer.step(model)
            lr_histories.append(optimizer.get_lr())

        plt.title("TransformerLRScheduler")
        plt.plot(lr_histories, label="lr", color="#FF6C38", linewidth=2)
        plt.legend()
        plt.grid(True)
        plt.xlabel("timestep", fontsize="large")
        plt.ylabel("lr", fontsize="large")
        plt.savefig("TransformerLRScheduler.png")

    def test_tri_stage_scheduler(self):
        configs = build_dummy_configs(scheduler_configs=TriStageLRSchedulerConfigs())

        lr_histories = list()
        total_steps = 15000

        model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]

        optimizer = optim.Adam(model, lr=1e-04)
        scheduler = TriStageLRScheduler(optimizer, configs)

        optimizer = Optimizer(
            optim=optimizer,
            scheduler=scheduler,
            scheduler_period=total_steps,
            max_grad_norm=0.0,
        )

        for timestep in range(total_steps):
            optimizer.step(model)
            lr_histories.append(optimizer.get_lr())

        plt.title("TransformerLRScheduler")
        plt.plot(lr_histories, label="lr", color="#FF6C38", linewidth=2)
        plt.legend()
        plt.grid(True)
        plt.xlabel("timestep", fontsize="large")
        plt.ylabel("lr", fontsize="large")
        plt.savefig("TransformerLRScheduler.png")

    def test_warmup_reduce_lr_on_plateau_scheduler(self):
        configs = build_dummy_configs(scheduler_configs=WarmupReduceLROnPlateauConfigs())

        lr_histories = list()
        total_steps = 15000

        model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]

        optimizer = optim.Adam(model, lr=1e-04)
        scheduler = WarmupReduceLROnPlateauScheduler(optimizer, configs)

        optimizer = Optimizer(
            optim=optimizer,
            scheduler=scheduler,
            scheduler_period=total_steps,
            max_grad_norm=0.0,
        )

        for timestep in range(total_steps):
            optimizer.step(model)
            lr_histories.append(optimizer.get_lr())

        plt.title("TransformerLRScheduler")
        plt.plot(lr_histories, label="lr", color="#FF6C38", linewidth=2)
        plt.legend()
        plt.grid(True)
        plt.xlabel("timestep", fontsize="large")
        plt.ylabel("lr", fontsize="large")
        plt.savefig("TransformerLRScheduler.png")


if __name__ == "__main__":
    unittest.main()
