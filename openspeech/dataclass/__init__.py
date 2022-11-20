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

from .configurations import (
    AIShellConfigs,
    AugmentConfigs,
    CPUResumeTrainerConfigs,
    CPUTrainerConfigs,
    EnsembleEvaluationConfigs,
    EvaluationConfigs,
    Fp16GPUTrainerConfigs,
    Fp16TPUTrainerConfigs,
    Fp64CPUTrainerConfigs,
    GPUResumeTrainerConfigs,
    GPUTrainerConfigs,
    KsponSpeechConfigs,
    LibriSpeechConfigs,
    LMConfigs,
    TPUResumeTrainerConfigs,
    TPUTrainerConfigs,
)

OPENSPEECH_TRAIN_CONFIGS = [
    "audio",
    "augment",
    "dataset",
    "model",
    "criterion",
    "lr_scheduler",
    "trainer",
    "tokenizer",
]

OPENSPEECH_LM_TRAIN_CONFIGS = [
    "dataset",
    "model",
    "criterion",
    "lr_scheduler",
    "trainer",
    "tokenizer",
]

DATASET_DATACLASS_REGISTRY = {
    "kspon": KsponSpeechConfigs,
    "libri": LibriSpeechConfigs,
    "aishell": AIShellConfigs,
    "ksponspeech": KsponSpeechConfigs,
    "librispeech": LibriSpeechConfigs,
    "lm": LMConfigs,
}
TRAINER_DATACLASS_REGISTRY = {
    "cpu": CPUTrainerConfigs,
    "gpu": GPUTrainerConfigs,
    "tpu": TPUTrainerConfigs,
    "gpu-fp16": Fp16GPUTrainerConfigs,
    "tpu-fp16": Fp16TPUTrainerConfigs,
    "cpu-fp64": Fp64CPUTrainerConfigs,
    "cpu-resume": CPUResumeTrainerConfigs,
    "gpu-resume": GPUResumeTrainerConfigs,
    "tpu-resume": TPUResumeTrainerConfigs,
}
AUGMENT_DATACLASS_REGISTRY = {
    "default": AugmentConfigs,
}
EVAL_DATACLASS_REGISTRY = {
    "default": EvaluationConfigs,
    "ensemble": EnsembleEvaluationConfigs,
}
