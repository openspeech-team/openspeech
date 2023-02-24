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
import os

from .openspeech_model import OpenspeechModel
from .openspeech_ctc_model import OpenspeechCTCModel
from .openspeech_encoder_decoder_model import OpenspeechEncoderDecoderModel
from .openspeech_transducer_model import OpenspeechTransducerModel

MODEL_REGISTRY = dict()
MODEL_DATACLASS_REGISTRY = dict()


def register_model(name: str, dataclass=None):
    r"""
    New model types can be added to OpenSpeech with the :func:`register_model` function decorator.

    For example::
        @register_model('conformer_lstm')
        class ConformerLSTMModel(OpenspeechModel):
            (...)

    .. note:: All models must implement the :class:`cls.__name__` interface.

    Args:
        name (str): the name of the model
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Cannot register duplicate model ({name})")
        if not issubclass(cls, OpenspeechModel):
            raise ValueError(f"Model ({name}: {cls.__name__}) must extend OpenspeechModel")

        MODEL_REGISTRY[name] = cls

        cls.__dataclass = dataclass
        if dataclass is not None:
            if name in MODEL_DATACLASS_REGISTRY:
                raise ValueError(f"Cannot register duplicate model ({name})")
            MODEL_DATACLASS_REGISTRY[name] = dataclass

        return cls

    return register_model_cls


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
for file in os.listdir(models_dir):
    if os.path.isdir(os.path.join(models_dir, file)) and not file.startswith("__"):
        for subfile in os.listdir(os.path.join(models_dir, file)):
            path = os.path.join(models_dir, file, subfile)
            if subfile.endswith(".py"):
                python_file = subfile[: subfile.find(".py")] if subfile.endswith(".py") else subfile
                module = importlib.import_module(f"openspeech.models.{file}.{python_file}")
        continue

    path = os.path.join(models_dir, file)
    if file.endswith(".py"):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(f"openspeech.models.{model_name}")


from .conformer import (
    ConformerConfigs,
    ConformerLSTMConfigs,
    ConformerLSTMModel,
    ConformerModel,
    ConformerTransducerConfigs,
    ConformerTransducerModel,
    JointCTCConformerLSTMConfigs,
    JointCTCConformerLSTMModel,
)
from .contextnet import (
    ContextNetConfigs,
    ContextNetLSTMConfigs,
    ContextNetLSTMModel,
    ContextNetModel,
    ContextNetTransducerConfigs,
    ContextNetTransducerModel,
)
from .deepspeech2 import DeepSpeech2Configs, DeepSpeech2Model
from .jasper import Jasper5x3Config, Jasper5x3Model, Jasper10x5Config, Jasper10x5Model
from .listen_attend_spell import (
    DeepCNNWithJointCTCListenAttendSpellConfigs,
    DeepCNNWithJointCTCListenAttendSpellModel,
    JointCTCListenAttendSpellConfigs,
    JointCTCListenAttendSpellModel,
    ListenAttendSpellConfigs,
    ListenAttendSpellModel,
    ListenAttendSpellWithLocationAwareConfigs,
    ListenAttendSpellWithLocationAwareModel,
    ListenAttendSpellWithMultiHeadConfigs,
    ListenAttendSpellWithMultiHeadModel,
)
from .lstm_lm import LSTMLanguageModel, LSTMLanguageModelConfigs
from .quartznet import (
    QuartzNet5x5Configs,
    QuartzNet5x5Model,
    QuartzNet10x5Configs,
    QuartzNet10x5Model,
    QuartzNet15x5Configs,
    QuartzNet15x5Model,
)
from .rnn_transducer import RNNTransducerConfigs, RNNTransducerModel
from .squeezeformer import (
    JointCTCSqueezeformerLSTMConfigs,
    JointCTCSqueezeformerLSTMModel,
    SqueezeformerConfigs,
    SqueezeformerLSTMConfigs,
    SqueezeformerLSTMModel,
    SqueezeformerModel,
    SqueezeformerTransducerConfigs,
    SqueezeformerTransducerModel,
)
from .transformer import (
    JointCTCTransformerConfigs,
    JointCTCTransformerModel,
    TransformerConfigs,
    TransformerModel,
    TransformerWithCTCConfigs,
    TransformerWithCTCModel,
    VGGTransformerConfigs,
    VGGTransformerModel,
)
from .transformer_lm import TransformerLanguageModel, TransformerLanguageModelConfigs
from .transformer_transducer import TransformerTransducerConfigs, TransformerTransducerModel

__all__ = [
    "OpenspeechModel",
    "OpenspeechEncoderDecoderModel",
    "OpenspeechTransducerModel",
    "OpenspeechCTCModel",
    "ConformerModel",
    "ConformerConfigs",
    "ConformerLSTMModel",
    "ConformerLSTMConfigs",
    "ConformerTransducerModel",
    "ConformerTransducerConfigs",
    "ContextNetModel",
    "ContextNetConfigs",
    "ContextNetLSTMModel",
    "ContextNetLSTMConfigs",
    "ContextNetTransducerModel",
    "ContextNetTransducerConfigs",
    "DeepCNNWithJointCTCListenAttendSpellModel",
    "DeepCNNWithJointCTCListenAttendSpellConfigs",
    "DeepSpeech2Model",
    "DeepSpeech2Configs",
    "Jasper5x3Model",
    "Jasper5x3Config",
    "Jasper10x5Model",
    "Jasper10x5Config",
    "JointCTCConformerLSTMModel",
    "JointCTCConformerLSTMConfigs",
    "JointCTCListenAttendSpellConfigs",
    "JointCTCTransformerConfigs",
    "JointCTCTransformerModel",
    "JointCTCListenAttendSpellModel",
    "ListenAttendSpellConfigs",
    "ListenAttendSpellWithMultiHeadConfigs",
    "ListenAttendSpellWithLocationAwareConfigs",
    "ListenAttendSpellModel",
    "ListenAttendSpellWithLocationAwareModel",
    "ListenAttendSpellWithMultiHeadModel",
    "VGGTransformerConfigs",
    "VGGTransformerModel",
    "QuartzNet15x5Configs",
    "QuartzNet10x5Configs",
    "QuartzNet5x5Configs",
    "QuartzNet15x5Model",
    "QuartzNet10x5Model",
    "QuartzNet5x5Model",
    "RNNTransducerConfigs",
    "RNNTransducerModel",
    "SqueezeformerConfigs",
    "SqueezeformerLSTMConfigs",
    "SqueezeformerTransducerConfigs",
    "JointCTCSqueezeformerLSTMConfigs",
    "SqueezeformerTransducerModel",
    "SqueezeformerLSTMModel",
    "SqueezeformerModel",
    "JointCTCSqueezeformerLSTMModel",
    "TransformerModel",
    "TransformerConfigs",
    "TransformerWithCTCConfigs",
    "TransformerTransducerConfigs",
    "TransformerTransducerModel",
    "TransformerWithCTCModel",
    "LSTMLanguageModel",
    "LSTMLanguageModelConfigs",
    "TransformerLanguageModel",
    "TransformerLanguageModelConfigs",
]
