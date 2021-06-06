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

import os
import importlib

VOCAB_REGISTRY = dict()
VOCAB_DATACLASS_REGISTRY = dict()


def register_vocab(name: str, dataclass=None):
    """
    New vocab types can be added to OpenSpeech with the :func:`register_vocab` function decorator.

    For example::
        @register_vocab('character')
        class KsponSpeechCharacterVocabulary:
            (...)

    .. note:: All vocabs must implement the :class:`cls.__name__` interface.

    Args:
        name (str): the name of the vocab
    """

    def register_vocab_cls(cls):
        if name in VOCAB_REGISTRY:
            raise ValueError(f"Cannot register duplicate vocab ({name})")

        VOCAB_REGISTRY[name] = cls

        cls.__dataclass = dataclass
        if dataclass is not None:
            if name in VOCAB_DATACLASS_REGISTRY:
                raise ValueError(f"Cannot register duplicate vocab ({name})")
            VOCAB_DATACLASS_REGISTRY[name] = dataclass

        return cls

    return register_vocab_cls


vocab_dir = os.path.dirname(__file__)
for file in os.listdir(vocab_dir):
    if os.path.isdir(os.path.join(vocab_dir, file)) and file != '__pycache__':
        for subfile in os.listdir(os.path.join(vocab_dir, file)):
            path = os.path.join(vocab_dir, file, subfile)
            if subfile.endswith(".py"):
                vocab_name = subfile[: subfile.find(".py")] if subfile.endswith(".py") else subfile
                module = importlib.import_module(f"openspeech.vocabs.{file}.{vocab_name}")
        continue

    path = os.path.join(vocab_dir, file)
    if file.endswith(".py"):
        vocab_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(f"openspeech.vocabs.{vocab_name}")
