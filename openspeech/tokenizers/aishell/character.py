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

import csv
from dataclasses import dataclass, field

from omegaconf import DictConfig

from openspeech.dataclass.configurations import TokenizerConfigs
from openspeech.tokenizers import register_tokenizer
from openspeech.tokenizers.tokenizer import Tokenizer


@dataclass
class AIShellCharacterTokenizerConfigs(TokenizerConfigs):
    unit: str = field(default="aishell_character", metadata={"help": "Unit of vocabulary."})
    vocab_path: str = field(
        default="../../../data_aishell/aishell_labels.csv", metadata={"help": "Path of vocabulary file."}
    )


@register_tokenizer("aishell_character", dataclass=AIShellCharacterTokenizerConfigs)
class AIShellCharacterTokenizer(Tokenizer):
    r"""
    Tokenizer class in Character-units for AISHELL.

    Args:
        configs (DictConfig): configuration set.
    """

    def __init__(self, configs: DictConfig):
        super(AIShellCharacterTokenizer, self).__init__()
        self.vocab_dict, self.id_dict = self.load_vocab(
            vocab_path=configs.tokenizer.vocab_path,
            encoding=configs.tokenizer.encoding,
        )
        self.labels = self.vocab_dict.keys()
        self.sos_token = configs.tokenizer.sos_token
        self.eos_token = configs.tokenizer.eos_token
        self.pad_token = configs.tokenizer.pad_token
        self.sos_id = int(self.vocab_dict[configs.tokenizer.sos_token])
        self.eos_id = int(self.vocab_dict[configs.tokenizer.eos_token])
        self.pad_id = int(self.vocab_dict[configs.tokenizer.pad_token])
        self.blank_id = int(self.vocab_dict[configs.tokenizer.blank_token])
        self.vocab_path = configs.tokenizer.vocab_path

    def __len__(self):
        return len(self.labels)

    def decode(self, labels):
        r"""
        Converts label to string.

        Args:
            labels (numpy.ndarray): number label

        Returns: sentence
            - **sentence** (str or list): symbol of labels
        """
        if len(labels.shape) == 1:
            sentence = str()
            for label in labels:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                    continue
                sentence += self.id_dict[label.item()]
            return sentence

        sentences = list()
        for batch in labels:
            sentence = str()
            for label in batch:
                if label.item() == self.eos_id:
                    break
                elif label.item() == self.blank_id:
                    continue
                sentence += self.id_dict[label.item()]
            sentences.append(sentence)
        return sentences

    def encode(self, sentence):
        label = str()

        for ch in sentence:
            try:
                label += str(self.vocab_dict[ch]) + " "
            except KeyError:
                continue

        return label[:-1]

    def load_vocab(self, vocab_path, encoding="utf-8"):
        r"""
        Provides char2id, id2char

        Args:
            vocab_path (str): csv file with character labels
            encoding (str): encoding method

        Returns: unit2id, id2unit
            - **unit2id** (dict): unit2id[unit] = id
            - **id2unit** (dict): id2unit[id] = unit
        """
        unit2id = dict()
        id2unit = dict()

        try:
            with open(vocab_path, "r", encoding=encoding) as f:
                labels = csv.reader(f, delimiter=",")
                next(labels)

                for row in labels:
                    unit2id[row[1]] = row[0]
                    id2unit[int(row[0])] = row[1]

            return unit2id, id2unit
        except IOError:
            raise IOError("Character label file (csv format) doesn`t exist : {0}".format(vocab_path))
