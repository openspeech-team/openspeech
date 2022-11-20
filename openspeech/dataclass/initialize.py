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

from hydra.core.config_store import ConfigStore


def hydra_train_init() -> None:
    r"""initialize ConfigStore for hydra-train"""
    from openspeech.criterion import CRITERION_DATACLASS_REGISTRY
    from openspeech.data import AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY
    from openspeech.dataclass import (
        AUGMENT_DATACLASS_REGISTRY,
        DATASET_DATACLASS_REGISTRY,
        OPENSPEECH_TRAIN_CONFIGS,
        TRAINER_DATACLASS_REGISTRY,
    )
    from openspeech.models import MODEL_DATACLASS_REGISTRY
    from openspeech.optim.scheduler import SCHEDULER_DATACLASS_REGISTRY
    from openspeech.tokenizers import TOKENIZER_DATACLASS_REGISTRY

    registries = {
        "audio": AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY,
        "augment": AUGMENT_DATACLASS_REGISTRY,
        "dataset": DATASET_DATACLASS_REGISTRY,
        "trainer": TRAINER_DATACLASS_REGISTRY,
        "model": MODEL_DATACLASS_REGISTRY,
        "criterion": CRITERION_DATACLASS_REGISTRY,
        "lr_scheduler": SCHEDULER_DATACLASS_REGISTRY,
        "tokenizer": TOKENIZER_DATACLASS_REGISTRY,
    }

    cs = ConfigStore.instance()

    for group in OPENSPEECH_TRAIN_CONFIGS:
        dataclass_registry = registries[group]

        for k, v in dataclass_registry.items():
            cs.store(group=group, name=k, node=v, provider="openspeech")


def hydra_lm_train_init() -> None:
    from openspeech.criterion import CRITERION_DATACLASS_REGISTRY
    from openspeech.dataclass import DATASET_DATACLASS_REGISTRY, OPENSPEECH_LM_TRAIN_CONFIGS, TRAINER_DATACLASS_REGISTRY
    from openspeech.models import MODEL_DATACLASS_REGISTRY
    from openspeech.optim.scheduler import SCHEDULER_DATACLASS_REGISTRY
    from openspeech.tokenizers import TOKENIZER_DATACLASS_REGISTRY

    registries = {
        "dataset": DATASET_DATACLASS_REGISTRY,
        "trainer": TRAINER_DATACLASS_REGISTRY,
        "model": MODEL_DATACLASS_REGISTRY,
        "criterion": CRITERION_DATACLASS_REGISTRY,
        "lr_scheduler": SCHEDULER_DATACLASS_REGISTRY,
        "tokenizer": TOKENIZER_DATACLASS_REGISTRY,
    }

    cs = ConfigStore.instance()

    for group in OPENSPEECH_LM_TRAIN_CONFIGS:
        dataclass_registry = registries[group]

        for k, v in dataclass_registry.items():
            cs.store(group=group, name=k, node=v, provider="openspeech")


def hydra_eval_init() -> None:
    from openspeech.data import AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY
    from openspeech.dataclass import EVAL_DATACLASS_REGISTRY
    from openspeech.models import MODEL_DATACLASS_REGISTRY
    from openspeech.tokenizers import TOKENIZER_DATACLASS_REGISTRY

    registries = {
        "audio": AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY,
        "eval": EVAL_DATACLASS_REGISTRY,
        "model": MODEL_DATACLASS_REGISTRY,
        "tokenizer": TOKENIZER_DATACLASS_REGISTRY,
    }

    cs = ConfigStore.instance()

    for group in registries.keys():
        dataclass_registry = registries[group]

        for k, v in dataclass_registry.items():
            cs.store(group=group, name=k, node=v, provider="openspeech")
