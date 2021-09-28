import unittest
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from openspeech.callbacks import CheckpointEveryNSteps
from openspeech.utils import is_apex_available, build_dummy_configs
from openspeech.dataclass.configurations import (
    CPUTrainerConfigs,
    GPUTrainerConfigs,
    GPUResumeTrainerConfigs,
    Fp16GPUTrainerConfigs,
    Fp16TPUTrainerConfigs,
    Fp64CPUTrainerConfigs,
    CPUResumeTrainerConfigs,
    TPUTrainerConfigs,
    TPUResumeTrainerConfigs,
)


class TestLightningTrainer(unittest.TestCase):
    def test_cpu_trainer(self):
        amp_backend = None

        configs = build_dummy_configs(trainer_configs=CPUTrainerConfigs())
        logger = WandbLogger()

        if hasattr(configs.trainer, "amp_backend"):
            amp_backend = "apex" if configs.trainer.amp_backend == "apex" and is_apex_available() else "native"

        pl.Trainer(
            accelerator=configs.trainer.accelerator,
            gpus=1 if "gpu" in configs.trainer.name else None,
            tpu_cores=None if "tpu" not in configs.trainer.name else configs.trainer.tpu_cores,
            precision=32 if not hasattr(configs.trainer, "precision") else configs.trainer.precision,
            accumulate_grad_batches=configs.trainer.accumulate_grad_batches,
            auto_select_gpus=None if "gpu" not in configs.trainer.name else configs.trainer.auto_select_gpus,
            check_val_every_n_epoch=configs.trainer.check_val_every_n_epoch,
            logger=logger,
            amp_backend=amp_backend,
            auto_scale_batch_size=configs.trainer.auto_scale_batch_size,
            max_epochs=configs.trainer.max_epochs,
            gradient_clip_val=configs.trainer.gradient_clip_val,
            resume_from_checkpoint=None if "resume" not in configs.trainer.name else configs.trainer.checkpoint_path,
            callbacks=[
                LearningRateMonitor(logging_interval='step'),
                CheckpointEveryNSteps(configs.trainer.save_checkpoint_n_steps)
            ],
        )

    def test_gpu_trainer(self):
        amp_backend = None

        configs = build_dummy_configs(trainer_configs=GPUTrainerConfigs())
        logger = WandbLogger()

        if hasattr(configs.trainer, "amp_backend"):
            amp_backend = "apex" if configs.trainer.amp_backend == "apex" and is_apex_available() else "native"

        trainer = pl.Trainer(
            accelerator=configs.trainer.accelerator,
            gpus=1 if "gpu" in configs.trainer.name else None,
            tpu_cores=None if "tpu" not in configs.trainer.name else configs.trainer.tpu_cores,
            precision=32 if not hasattr(configs.trainer, "precision") else configs.trainer.precision,
            accumulate_grad_batches=configs.trainer.accumulate_grad_batches,
            auto_select_gpus=None if "gpu" not in configs.trainer.name else configs.trainer.auto_select_gpus,
            check_val_every_n_epoch=configs.trainer.check_val_every_n_epoch,
            logger=logger,
            amp_backend=amp_backend,
            auto_scale_batch_size=configs.trainer.auto_scale_batch_size,
            max_epochs=configs.trainer.max_epochs,
            gradient_clip_val=configs.trainer.gradient_clip_val,
            resume_from_checkpoint=None if "resume" not in configs.trainer.name else configs.trainer.checkpoint_path,
            callbacks=[
                LearningRateMonitor(logging_interval='step'),
                CheckpointEveryNSteps(configs.trainer.save_checkpoint_n_steps)
            ],
        )

    def test_tpu_trainer(self):
        amp_backend = None

        configs = build_dummy_configs(trainer_configs=TPUTrainerConfigs())
        logger = WandbLogger()

        if hasattr(configs.trainer, "amp_backend"):
            amp_backend = "apex" if configs.trainer.amp_backend == "apex" and is_apex_available() else "native"

        trainer = pl.Trainer(
            accelerator=configs.trainer.accelerator,
            gpus=1 if "gpu" in configs.trainer.name else None,
            tpu_cores=None if "tpu" not in configs.trainer.name else configs.trainer.tpu_cores,
            precision=32 if not hasattr(configs.trainer, "precision") else configs.trainer.precision,
            accumulate_grad_batches=configs.trainer.accumulate_grad_batches,
            auto_select_gpus=None if "gpu" not in configs.trainer.name else configs.trainer.auto_select_gpus,
            check_val_every_n_epoch=configs.trainer.check_val_every_n_epoch,
            logger=logger,
            amp_backend=amp_backend,
            auto_scale_batch_size=configs.trainer.auto_scale_batch_size,
            max_epochs=configs.trainer.max_epochs,
            gradient_clip_val=configs.trainer.gradient_clip_val,
            resume_from_checkpoint=None if "resume" not in configs.trainer.name else configs.trainer.checkpoint_path,
            callbacks=[
                LearningRateMonitor(logging_interval='step'),
                CheckpointEveryNSteps(configs.trainer.save_checkpoint_n_steps)
            ],
        )

    def test_cpu_resume_trainer(self):
        amp_backend = None

        configs = build_dummy_configs(trainer_configs=CPUResumeTrainerConfigs())
        logger = WandbLogger()

        if hasattr(configs.trainer, "amp_backend"):
            amp_backend = "apex" if configs.trainer.amp_backend == "apex" and is_apex_available() else "native"

        trainer = pl.Trainer(
            accelerator=configs.trainer.accelerator,
            gpus=1 if "gpu" in configs.trainer.name else None,
            tpu_cores=None if "tpu" not in configs.trainer.name else configs.trainer.tpu_cores,
            precision=32 if not hasattr(configs.trainer, "precision") else configs.trainer.precision,
            accumulate_grad_batches=configs.trainer.accumulate_grad_batches,
            auto_select_gpus=None if "gpu" not in configs.trainer.name else configs.trainer.auto_select_gpus,
            check_val_every_n_epoch=configs.trainer.check_val_every_n_epoch,
            logger=logger,
            amp_backend=amp_backend,
            auto_scale_batch_size=configs.trainer.auto_scale_batch_size,
            max_epochs=configs.trainer.max_epochs,
            gradient_clip_val=configs.trainer.gradient_clip_val,
            resume_from_checkpoint=None if "resume" not in configs.trainer.name else configs.trainer.checkpoint_path,
            callbacks=[
                LearningRateMonitor(logging_interval='step'),
                CheckpointEveryNSteps(configs.trainer.save_checkpoint_n_steps)
            ],
        )

    def test_gpu_resume_trainer(self):
        amp_backend = None

        configs = build_dummy_configs(trainer_configs=GPUResumeTrainerConfigs())
        logger = WandbLogger()

        if hasattr(configs.trainer, "amp_backend"):
            amp_backend = "apex" if configs.trainer.amp_backend == "apex" and is_apex_available() else "native"

        trainer = pl.Trainer(
            accelerator=configs.trainer.accelerator,
            gpus=1 if "gpu" in configs.trainer.name else None,
            tpu_cores=None if "tpu" not in configs.trainer.name else configs.trainer.tpu_cores,
            precision=32 if not hasattr(configs.trainer, "precision") else configs.trainer.precision,
            accumulate_grad_batches=configs.trainer.accumulate_grad_batches,
            auto_select_gpus=None if "gpu" not in configs.trainer.name else configs.trainer.auto_select_gpus,
            check_val_every_n_epoch=configs.trainer.check_val_every_n_epoch,
            logger=logger,
            amp_backend=amp_backend,
            auto_scale_batch_size=configs.trainer.auto_scale_batch_size,
            max_epochs=configs.trainer.max_epochs,
            gradient_clip_val=configs.trainer.gradient_clip_val,
            resume_from_checkpoint=None if "resume" not in configs.trainer.name else configs.trainer.checkpoint_path,
            callbacks=[
                LearningRateMonitor(logging_interval='step'),
                CheckpointEveryNSteps(configs.trainer.save_checkpoint_n_steps)
            ],
        )

    def test_tpu_resume_trainer(self):
        amp_backend = None

        configs = build_dummy_configs(trainer_configs=TPUResumeTrainerConfigs())
        logger = WandbLogger()

        if hasattr(configs.trainer, "amp_backend"):
            amp_backend = "apex" if configs.trainer.amp_backend == "apex" and is_apex_available() else "native"

        trainer = pl.Trainer(
            accelerator=configs.trainer.accelerator,
            gpus=1 if "gpu" in configs.trainer.name else None,
            tpu_cores=None if "tpu" not in configs.trainer.name else configs.trainer.tpu_cores,
            precision=32 if not hasattr(configs.trainer, "precision") else configs.trainer.precision,
            accumulate_grad_batches=configs.trainer.accumulate_grad_batches,
            auto_select_gpus=None if "gpu" not in configs.trainer.name else configs.trainer.auto_select_gpus,
            check_val_every_n_epoch=configs.trainer.check_val_every_n_epoch,
            logger=logger,
            amp_backend=amp_backend,
            auto_scale_batch_size=configs.trainer.auto_scale_batch_size,
            max_epochs=configs.trainer.max_epochs,
            gradient_clip_val=configs.trainer.gradient_clip_val,
            resume_from_checkpoint=None if "resume" not in configs.trainer.name else configs.trainer.checkpoint_path,
            callbacks=[
                LearningRateMonitor(logging_interval='step'),
                CheckpointEveryNSteps(configs.trainer.save_checkpoint_n_steps)
            ],
        )

    def test_fp64_cpu_resume_trainer(self):
        amp_backend = None

        configs = build_dummy_configs(trainer_configs=Fp64CPUTrainerConfigs())
        logger = WandbLogger()

        if hasattr(configs.trainer, "amp_backend"):
            amp_backend = "apex" if configs.trainer.amp_backend == "apex" and is_apex_available() else "native"

        trainer = pl.Trainer(
            accelerator=configs.trainer.accelerator,
            gpus=1 if "gpu" in configs.trainer.name else None,
            tpu_cores=None if "tpu" not in configs.trainer.name else configs.trainer.tpu_cores,
            precision=32 if not hasattr(configs.trainer, "precision") else configs.trainer.precision,
            accumulate_grad_batches=configs.trainer.accumulate_grad_batches,
            auto_select_gpus=None if "gpu" not in configs.trainer.name else configs.trainer.auto_select_gpus,
            check_val_every_n_epoch=configs.trainer.check_val_every_n_epoch,
            logger=logger,
            amp_backend=amp_backend,
            auto_scale_batch_size=configs.trainer.auto_scale_batch_size,
            max_epochs=configs.trainer.max_epochs,
            gradient_clip_val=configs.trainer.gradient_clip_val,
            resume_from_checkpoint=None if "resume" not in configs.trainer.name else configs.trainer.checkpoint_path,
            callbacks=[
                LearningRateMonitor(logging_interval='step'),
                CheckpointEveryNSteps(configs.trainer.save_checkpoint_n_steps)
            ],
        )

    def test_fp16_gpu_resume_trainer(self):
        amp_backend = None

        configs = build_dummy_configs(trainer_configs=Fp16GPUTrainerConfigs())
        logger = WandbLogger()

        if hasattr(configs.trainer, "amp_backend"):
            amp_backend = "apex" if configs.trainer.amp_backend == "apex" and is_apex_available() else "native"

        trainer = pl.Trainer(
            accelerator=configs.trainer.accelerator,
            gpus=1 if "gpu" in configs.trainer.name else None,
            tpu_cores=None if "tpu" not in configs.trainer.name else configs.trainer.tpu_cores,
            precision=32 if not hasattr(configs.trainer, "precision") else configs.trainer.precision,
            accumulate_grad_batches=configs.trainer.accumulate_grad_batches,
            auto_select_gpus=None if "gpu" not in configs.trainer.name else configs.trainer.auto_select_gpus,
            check_val_every_n_epoch=configs.trainer.check_val_every_n_epoch,
            logger=logger,
            amp_backend=amp_backend,
            auto_scale_batch_size=configs.trainer.auto_scale_batch_size,
            max_epochs=configs.trainer.max_epochs,
            gradient_clip_val=configs.trainer.gradient_clip_val,
            resume_from_checkpoint=None if "resume" not in configs.trainer.name else configs.trainer.checkpoint_path,
            callbacks=[
                LearningRateMonitor(logging_interval='step'),
                CheckpointEveryNSteps(configs.trainer.save_checkpoint_n_steps)
            ],
        )

    def test_fp16_tpu_resume_trainer(self):
        amp_backend = None

        configs = build_dummy_configs(trainer_configs=Fp16TPUTrainerConfigs())
        logger = WandbLogger()

        if hasattr(configs.trainer, "amp_backend"):
            amp_backend = "apex" if configs.trainer.amp_backend == "apex" and is_apex_available() else "native"

        trainer = pl.Trainer(
            accelerator=configs.trainer.accelerator,
            gpus=1 if "gpu" in configs.trainer.name else None,
            tpu_cores=None if "tpu" not in configs.trainer.name else configs.trainer.tpu_cores,
            precision=32 if not hasattr(configs.trainer, "precision") else configs.trainer.precision,
            accumulate_grad_batches=configs.trainer.accumulate_grad_batches,
            auto_select_gpus=None if "gpu" not in configs.trainer.name else configs.trainer.auto_select_gpus,
            check_val_every_n_epoch=configs.trainer.check_val_every_n_epoch,
            logger=logger,
            amp_backend=amp_backend,
            auto_scale_batch_size=configs.trainer.auto_scale_batch_size,
            max_epochs=configs.trainer.max_epochs,
            gradient_clip_val=configs.trainer.gradient_clip_val,
            resume_from_checkpoint=None if "resume" not in configs.trainer.name else configs.trainer.checkpoint_path,
            callbacks=[
                LearningRateMonitor(logging_interval='step'),
                CheckpointEveryNSteps(configs.trainer.save_checkpoint_n_steps)
            ],
        )
