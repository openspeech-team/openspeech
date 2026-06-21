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

import importlib
import logging
import platform
from collections import OrderedDict
from typing import Iterable, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import LearningRateMonitor

from .callbacks import CheckpointEveryNSteps

PYTORCH_IMPORT_ERROR = """
Openspeech requires the PyTorch library but it was not found in your environment. Checkout the instructions on the
installation page: https://pytorch.org/get-started/locally/ and follow the ones that match your environment.
"""

TORCHAUDIO_IMPORT_ERROR = """
Openspeech requires the torchaudio library but it was not found in your environment. You can install it with pip:
`pip install torchaudio`
"""

PYTORCH_LIGHTNING_IMPORT_ERROR = """
Openspeech requires the lightning library but it was not found in your environment. You can install it with pip:
`pip install lightning`
"""

HYDRA_IMPORT_ERROR = """
Openspeech requires the hydra library but it was not found in your environment. You can install it with pip:
`pip install hydra-core`
"""

LIBROSA_IMPORT_ERROR = """
Openspeech requires the librosa library but it was not found in your environment. You can install it with pip:
`pip install librosa`
"""

SENTENCEPIECE_IMPORT_ERROR = """
Openspeech requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment.
"""

WARPRNNT_IMPORT_ERROR = """
Openspeech requires the warp-rnnt library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/1ytic/warp-rnnt and follow the ones that match your environment.
"""

CTCDECODE_IMPORT_ERROR = """
Openspeech requires the ctcdecode library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/parlance/ctcdecode and follow the ones that match your environment.
"""

try:
    import librosa
except ImportError:
    raise ValueError(LIBROSA_IMPORT_ERROR)

try:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger
except ImportError:
    raise ValueError(PYTORCH_LIGHTNING_IMPORT_ERROR)

try:
    DUMMY_SIGNALS, _ = librosa.load(librosa.ex("choice"))
except Exception:
    # `librosa.ex` downloads a sample over the network; fall back to a synthetic
    # signal so that `import openspeech` works offline / in CI.
    import numpy as _np

    DUMMY_SIGNALS = _np.zeros(22050, dtype="float32")
DUMMY_FEATURES = librosa.feature.melspectrogram(y=DUMMY_SIGNALS, n_mels=80)
DUMMY_INPUTS = torch.FloatTensor(DUMMY_FEATURES).transpose(0, 1).unsqueeze(0).expand(3, -1, -1)
DUMMY_INPUT_LENGTHS = torch.IntTensor([1070, 900, 800])
DUMMY_TARGETS = torch.LongTensor(
    [
        [2, 3, 3, 3, 3, 3, 2, 2, 1, 0],
        [2, 3, 3, 3, 3, 3, 2, 1, 2, 0],
        [2, 3, 3, 3, 3, 3, 2, 2, 0, 1],
    ]
)
DUMMY_TARGET_LENGTHS = torch.IntTensor([9, 8, 7])
DUMMY_TRANSCRIPTS = "OPENSPEECH IS AWESOME"

DUMMY_LM_INPUTS = torch.LongTensor(
    [
        [2, 3, 3, 3, 3, 3, 2, 2, 0],
        [2, 3, 3, 3, 3, 3, 2, 3, 2],
        [2, 3, 3, 3, 3, 3, 2, 2, 0],
    ]
)
DYMMY_LM_INPUT_LENGTHS = torch.IntTensor([9, 8, 7])
DUMMY_LM_TARGETS = torch.LongTensor(
    [
        [3, 3, 3, 3, 3, 2, 2, 1, 0],
        [3, 3, 3, 3, 3, 2, 1, 2, 0],
        [3, 3, 3, 3, 3, 2, 2, 0, 1],
    ]
)


def is_pytorch_available():
    return importlib.util.find_spec("torch") is not None


def is_pytorch_lightning_available():
    return importlib.util.find_spec("lightning") is not None


def is_hydra_available():
    return importlib.util.find_spec("hydra") is not None


def is_librosa_available():
    return importlib.util.find_spec("librosa") is not None


def is_apex_available():
    return importlib.util.find_spec("apex") is not None


def is_sentencepiece_available():
    return importlib.util.find_spec("sentencepiece") is not None


def is_torchaudio_available():
    return importlib.util.find_spec("torchaudio") is not None


BACKENDS_MAPPING = OrderedDict(
    [
        ("torch", (is_pytorch_available, PYTORCH_IMPORT_ERROR)),
        ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
        ("pytorch_lightning", (is_pytorch_lightning_available, PYTORCH_LIGHTNING_IMPORT_ERROR)),
        ("hydra", (is_hydra_available, HYDRA_IMPORT_ERROR)),
        ("librosa", (is_librosa_available, LIBROSA_IMPORT_ERROR)),
        ("torchaudio", (is_torchaudio_available, TORCHAUDIO_IMPORT_ERROR)),
    ]
)


def check_backends():
    backends = BACKENDS_MAPPING.keys()

    if not all(BACKENDS_MAPPING[backend][0]() for backend in backends):
        raise ImportError("".join([BACKENDS_MAPPING[backend][1] for backend in backends]))


def get_class_name(obj):
    return obj.__class__.__name__


def _check_environment(use_cuda: bool, logger) -> int:
    r"""
    Check execution envirionment.
    OS, Processor, CUDA version, Pytorch version, ... etc.
    """
    check_backends()

    cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    logger.info(f"Operating System : {platform.system()} {platform.release()}")
    logger.info(f"Processor : {platform.processor()}")

    num_devices = torch.cuda.device_count()

    if str(device) == "cuda":
        for idx in range(torch.cuda.device_count()):
            logger.info(f"device : {torch.cuda.get_device_name(idx)}")
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"CUDA version : {torch.version.cuda}")
        logger.info(f"PyTorch version : {torch.__version__}")

    else:
        logger.info(f"CUDA is available : {torch.cuda.is_available()}")
        logger.info(f"PyTorch version : {torch.__version__}")

    return num_devices


def parse_configs(configs: DictConfig) -> Tuple[Union[TensorBoardLogger, bool], int]:
    r"""
    Parsing configuration set.

    Args:
        configs (DictConfig): configuration set.

    Returns:
        logger (Union[TensorBoardLogger, bool]): logger for training
        num_devices (int): the number of cuda device
    """
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(configs))
    num_devices = _check_environment(configs.trainer.use_cuda, logger)

    if configs.trainer.logger == "tensorboard":
        logger = TensorBoardLogger("logs/")
    elif configs.trainer.logger == "wandb":
        logger = WandbLogger(
            project=f"{configs.model.model_name}-{configs.dataset.dataset}",
            name=f"{configs.model.model_name}-{configs.dataset.dataset}",
            job_type="train",
        )
    else:
        logger = True

    return logger, num_devices


def resolve_accelerator_settings(name: str, num_devices: int, tpu_cores: int = 8) -> dict:
    r"""
    Map a ``trainer.name`` onto Lightning 2.x ``accelerator`` / ``devices`` / ``precision``.

    The legacy per-device Trainer args (``gpus``, ``tpu_cores``, ``auto_select_gpus``,
    ``auto_scale_batch_size``, ``amp_backend``, ``resume_from_checkpoint``) were removed in
    Lightning 2.x. This pure helper centralizes the translation so it can be unit-tested
    without constructing a real ``Trainer``.

    Args:
        name (str): trainer name, one of cpu/gpu/tpu optionally suffixed with -fp16/-fp64/-resume.
        num_devices (int): number of available CUDA devices (used only for the gpu accelerator).
        tpu_cores (int): number of TPU cores (used only for the tpu accelerator).

    Returns:
        dict: ``{"accelerator": str, "devices": int | str, "precision": str}``
    """
    if name.startswith("cpu"):
        accelerator = "cpu"
    elif name.startswith("gpu"):
        accelerator = "gpu"
    elif name.startswith("tpu"):
        accelerator = "tpu"
    else:
        raise ValueError(f"Unsupported trainer: {name}")

    if name.endswith("fp16"):
        precision = "16-mixed"
    elif name.endswith("fp64"):
        precision = "64-true"
    else:
        precision = "32-true"

    if accelerator == "gpu":
        devices = num_devices if num_devices and num_devices > 0 else "auto"
    elif accelerator == "tpu":
        devices = tpu_cores
    else:
        devices = "auto"

    return {"accelerator": accelerator, "devices": devices, "precision": precision}


def get_pl_trainer(configs: DictConfig, num_devices: int, logger: Union[Logger, Iterable[Logger], bool]) -> pl.Trainer:
    r"""
    Build a PyTorch-Lightning 2.x ``Trainer`` from the trainer configs.

    Hardware/precision settings come from :func:`resolve_accelerator_settings`; the distributed
    ``strategy`` is read from the trainer config (default ``"auto"``). Checkpoint resuming is
    handled by passing ``ckpt_path`` to ``trainer.fit(...)`` in the CLI.
    """
    settings = resolve_accelerator_settings(configs.trainer.name, num_devices, getattr(configs.trainer, "tpu_cores", 8))
    strategy = getattr(configs.trainer, "strategy", "auto") or "auto"

    devices = settings["devices"]
    devices_override = getattr(configs.trainer, "devices", "auto")
    if devices_override not in (None, "auto"):
        devices = int(devices_override) if str(devices_override).isdigit() else devices_override

    return pl.Trainer(
        accelerator=settings["accelerator"],
        devices=devices,
        strategy=strategy,
        precision=settings["precision"],
        accumulate_grad_batches=configs.trainer.accumulate_grad_batches,
        check_val_every_n_epoch=configs.trainer.check_val_every_n_epoch,
        gradient_clip_val=configs.trainer.gradient_clip_val,
        logger=logger,
        max_epochs=configs.trainer.max_epochs,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            CheckpointEveryNSteps(configs.trainer.save_checkpoint_n_steps),
        ],
    )


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def build_dummy_configs(
    model_configs=None,
    vocab_configs=None,
    criterion_configs=None,
    scheduler_configs=None,
    trainer_configs=None,
    audio_configs=None,
):
    from openspeech.criterion import CrossEntropyLossConfigs
    from openspeech.data.audio.melspectrogram.melspectrogram import MelSpectrogramConfigs
    from openspeech.dataclass import CPUTrainerConfigs
    from openspeech.models import ConformerConfigs
    from openspeech.optim.scheduler.warmup_reduce_lr_on_plateau_scheduler import WarmupReduceLROnPlateauConfigs
    from openspeech.tokenizers.ksponspeech.character import KsponSpeechCharacterTokenizerConfigs

    if model_configs is None:
        model_configs = ConformerConfigs()

    if vocab_configs is None:
        vocab_configs = KsponSpeechCharacterTokenizerConfigs()
        vocab_configs.vocab_path = "labels.csv"

    if criterion_configs is None:
        criterion_configs = CrossEntropyLossConfigs

    if trainer_configs is None:
        trainer_configs = CPUTrainerConfigs()

    if scheduler_configs is None:
        scheduler_configs = WarmupReduceLROnPlateauConfigs()

    if audio_configs is None:
        audio_configs = MelSpectrogramConfigs()

    return DotDict(
        {
            "model": model_configs,
            "vocab": vocab_configs,
            "criterion": criterion_configs,
            "trainer": trainer_configs,
            "audio": audio_configs,
            "lr_scheduler": scheduler_configs,
        }
    )
