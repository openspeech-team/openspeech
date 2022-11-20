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

DATA_MODULE_REGISTRY = dict()


def register_data_module(name: str):
    """
    New data module types can be added to OpenSpeech with the :func:`register_data_module` function decorator.

    For example::
        @register_data_module('ksponspeech')
        class LightningKsponSpeechDataModule:
            (...)

    .. note:: All vocabs must implement the :class:`cls.__name__` interface.

    Args:
        name (str): the name of the vocab
    """

    def register_data_module_cls(cls):
        if name in DATA_MODULE_REGISTRY:
            raise ValueError(f"Cannot register duplicate data module ({name})")
        DATA_MODULE_REGISTRY[name] = cls
        return cls

    return register_data_module_cls


data_module_dir = os.path.dirname(__file__)
for file in os.listdir(data_module_dir):
    if os.path.isdir(os.path.join(data_module_dir, file)) and file != "__pycache__":
        for subfile in os.listdir(os.path.join(data_module_dir, file)):
            path = os.path.join(data_module_dir, file, subfile)
            if subfile.endswith(".py"):
                data_module_name = subfile[: subfile.find(".py")] if subfile.endswith(".py") else subfile
                module = importlib.import_module(f"openspeech.datasets.{file}.{data_module_name}")
