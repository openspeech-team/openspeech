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

AUDIO_FEATURE_TRANSFORM_REGISTRY = dict()
AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY = dict()


def register_audio_feature_transform(name: str, dataclass=None):
    r"""
    New dataset types can be added to OpenSpeech with the :func:`register_dataset` function decorator.

    For example::
        @register_audio_feature_transform("fbank", dataclass=FilterBankConfigs)
        class FilterBankFeatureTransform(object):
            (...)

    .. note:: All dataset must implement the :class:`cls.__name__` interface.

    Args:
        name (str): the name of the dataset
        dataclass (Optional, str): the dataclass of the dataset (default: None)
    """

    def register_audio_feature_transform_cls(cls):
        if name in AUDIO_FEATURE_TRANSFORM_REGISTRY:
            raise ValueError(f"Cannot register duplicate audio ({name})")

        AUDIO_FEATURE_TRANSFORM_REGISTRY[name] = cls

        cls.__dataclass = dataclass
        if dataclass is not None:
            if name in AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY:
                raise ValueError(f"Cannot register duplicate dataclass ({name})")
            AUDIO_FEATURE_TRANSFORM_DATACLASS_REGISTRY[name] = dataclass

        return cls

    return register_audio_feature_transform_cls


data_dir = os.path.dirname(__file__)
for file in os.listdir(f"{data_dir}/audio"):
    if os.path.isdir(f"{data_dir}/audio/{file}") and not file.startswith("__"):
        path = f"{data_dir}/audio/{file}"
        for module_file in os.listdir(path):
            path = os.path.join(path, module_file)
            if module_file.endswith(".py"):
                module_name = module_file[: module_file.find(".py")] if module_file.endswith(".py") else module_file
                module = importlib.import_module(f"openspeech.data.audio.{file}.{module_name}")
