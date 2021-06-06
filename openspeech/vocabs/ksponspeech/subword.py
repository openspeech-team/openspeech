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

import sentencepiece as spm
from dataclasses import dataclass, MISSING, field
from omegaconf import DictConfig

from openspeech.dataclass.configurations import VocabularyConfigs
from openspeech.vocabs import register_vocab
from openspeech.vocabs.vocab import Vocabulary


@dataclass
class KsponSpeechSubwordVocabConfigs(VocabularyConfigs):
    unit: str = field(
        default="kspon_subword", metadata={"help": "Unit of vocabulary."}
    )
    sp_model_path: str = field(
        default="sp.model", metadata={"help": "Path of sentencepiece model."}
    )
    sos_token: str = field(
        default="<s>", metadata={"help": "Start of sentence token"}
    )
    eos_token: str = field(
        default="</s>", metadata={"help": "End of sentence token"}
    )
    vocab_size: int = field(
        default=3200, metadata={"help": "Size of vocabulary."}
    )


@register_vocab("kspon_subword", dataclass=KsponSpeechSubwordVocabConfigs)
class KsponSpeechSubwordVocabulary(Vocabulary):
    """
    Vocabulary Class in Subword Units.

    Args:
        configs (DictConfig): configuration set.
    """
    def __init__(self, configs: DictConfig):
        super(KsponSpeechSubwordVocabulary, self).__init__()
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(configs.vocab.sp_model_path)

        self.vocab_dict = [[self.sp.id_to_piece(id), id] for id in range(self.sp.get_piece_size())]
        self.labels = [item[0] for item in self.vocab_dict]

        self.pad_id = self.sp.PieceToId(configs.vocab.pad_token)
        self.sos_id = self.sp.PieceToId(configs.vocab.sos_token)
        self.eos_id = self.sp.PieceToId(configs.vocab.eos_token)
        self.blank_id = self.sp.PieceToId(configs.vocab.blank_token)
        self.vocab_size = configs.vocab.vocab_size

    def __len__(self):
        return self.vocab_size

    def label_to_string(self, labels):
        """
        Converts label to string (number => Hangeul)

        Args:
            labels (numpy.ndarray): number label

        Returns: sentence
            - **sentence** (str or list): symbol of labels
        """
        if len(labels.shape) == 1:
            return self.sp.DecodeIds([int(l) for l in labels])

        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                sentence = self.sp.DecodeIds([int(l) for l in label])
            sentences.append(sentence)
        return sentences
