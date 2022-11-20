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
from dataclasses import dataclass, field

from omegaconf import DictConfig

from openspeech.dataclass.configurations import TokenizerConfigs
from openspeech.datasets.librispeech.preprocess.subword import SENTENCEPIECE_MODEL_NAME
from openspeech.tokenizers import register_tokenizer
from openspeech.tokenizers.tokenizer import Tokenizer
from openspeech.utils import SENTENCEPIECE_IMPORT_ERROR


@dataclass
class LibriSpeechSubwordTokenizerConfigs(TokenizerConfigs):
    unit: str = field(default="libri_subword", metadata={"help": "Unit of vocabulary."})
    sos_token: str = field(default="<s>", metadata={"help": "Start of sentence token"})
    eos_token: str = field(default="</s>", metadata={"help": "End of sentence token"})
    vocab_size: int = field(default=5000, metadata={"help": "Size of vocabulary."})
    vocab_path: str = field(default="../../../LibriSpeech/", metadata={"help": "Path of vocabulary file."})


@register_tokenizer("libri_subword", dataclass=LibriSpeechSubwordTokenizerConfigs)
class LibriSpeechSubwordTokenizer(Tokenizer):
    """
    Tokenizer class in Subword-units for LibriSpeech.

    Args:
        configs (DictConfig): configuration set.
    """

    def __init__(self, configs: DictConfig):
        super(LibriSpeechSubwordTokenizer, self).__init__()
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError(SENTENCEPIECE_IMPORT_ERROR)

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(os.path.join(configs.tokenizer.vocab_path, f"{SENTENCEPIECE_MODEL_NAME}.model"))
        self.pad_id = self.sp.PieceToId(configs.tokenizer.pad_token)
        self.sos_id = self.sp.PieceToId(configs.tokenizer.sos_token)
        self.eos_id = self.sp.PieceToId(configs.tokenizer.eos_token)
        self.blank_id = self.sp.PieceToId(configs.tokenizer.blank_token)
        self.vocab_size = configs.tokenizer.vocab_size

    def __len__(self):
        return self.vocab_size

    def decode(self, labels):
        if len(labels.shape) == 1:
            return self.sp.DecodeIds([l.item() for l in labels])

        elif len(labels.shape) == 2:
            sentences = list()

            for label in labels:
                sentence = self.sp.DecodeIds([l.item() for l in label])
                sentences.append(sentence)
            return sentences
        else:
            raise ValueError("Unsupported label's shape")

    def encode(self, sentence):
        text = " ".join(self.sp.EncodeAsPieces(sentence))
        label = " ".join([str(self.sp.PieceToId(token)) for token in text])
        return label
