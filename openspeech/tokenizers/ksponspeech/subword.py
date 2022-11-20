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

from dataclasses import dataclass, field

import sentencepiece as spm
from omegaconf import DictConfig

from openspeech.dataclass.configurations import TokenizerConfigs
from openspeech.tokenizers import register_tokenizer
from openspeech.tokenizers.tokenizer import Tokenizer


@dataclass
class KsponSpeechSubwordTokenizerConfigs(TokenizerConfigs):
    unit: str = field(default="kspon_subword", metadata={"help": "Unit of vocabulary."})
    sp_model_path: str = field(default="sp.model", metadata={"help": "Path of sentencepiece model."})
    sos_token: str = field(default="<s>", metadata={"help": "Start of sentence token"})
    eos_token: str = field(default="</s>", metadata={"help": "End of sentence token"})
    vocab_size: int = field(default=3200, metadata={"help": "Size of vocabulary."})


@register_tokenizer("kspon_subword", dataclass=KsponSpeechSubwordTokenizerConfigs)
class KsponSpeechSubwordTokenizer(Tokenizer):
    """
    Tokenizer class in Subword-units for KsponSpeech.

    Args:
        configs (DictConfig): configuration set.
    """

    def __init__(self, configs: DictConfig):
        super(KsponSpeechSubwordTokenizer, self).__init__()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(configs.tokenizer.sp_model_path)

        self.vocab_dict = [[self.sp.id_to_piece(id), id] for id in range(self.sp.get_piece_size())]
        self.labels = [item[0] for item in self.vocab_dict]

        self.pad_id = self.sp.PieceToId(configs.tokenizer.pad_token)
        self.sos_id = self.sp.PieceToId(configs.tokenizer.sos_token)
        self.eos_id = self.sp.PieceToId(configs.tokenizer.eos_token)
        self.blank_id = self.sp.PieceToId(configs.tokenizer.blank_token)
        self.vocab_size = configs.tokenizer.vocab_size

    def __len__(self):
        return self.vocab_size

    def decode(self, labels):
        """
        Converts label to string (number => Hangeul)

        Args:
            labels (numpy.ndarray): number label

        Returns: sentence
            - **sentence** (str or list): symbol of labels
        """
        if len(labels.shape) == 1:
            return self.sp.DecodeIds([int(_) for _ in labels])

        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                sentence = self.sp.DecodeIds([int(_) for _ in label])
            sentences.append(sentence)
        return sentences

    def encode(self, sentence):
        text = " ".join(self.sp.EncodeAsPieces(sentence))
        label = " ".join([str(self.sp.PieceToId(token)) for token in text])
        return label
