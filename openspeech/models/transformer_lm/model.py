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

import torch
from omegaconf import DictConfig
from collections import OrderedDict
from typing import Dict

from openspeech.lm.transformer_lm import TransformerForLanguageModel
from openspeech.models import register_model, OpenspeechModel
from openspeech.models.transformer_lm.configurations import TransformerLanguageModelConfigs
from openspeech.vocabs.vocab import Vocabulary


@register_model('transformer_lm', dataclass=TransformerLanguageModelConfigs)
class TransformerLanguageModel(OpenspeechModel):
    r"""
    Transformer language model.
    Paper: https://arxiv.org/abs/1904.09408

    Args:
        configs (DictConfig): configuration set.
        vocab (Vocabulary): the class of vocabulary

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be
            a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        * outputs (dict): Result of model predictions.
    """
    def __init__(self, configs: DictConfig, vocab: Vocabulary, ) -> None:
        super(TransformerLanguageModel, self).__init__(configs, vocab)

    def build_model(self):
        self.lm = TransformerForLanguageModel(
            num_classes=self.num_classes,
            max_length=self.configs.model.max_length,
            d_model=self.configs.model.d_model,
            d_ff=self.configs.model.d_ff,
            num_attention_heads=self.configs.model.num_attention_heads,
            pad_id=self.vocab.pad_id,
            sos_id=self.vocab.sos_id,
            eos_id=self.vocab.eos_id,
            dropout_p=self.configs.model.dropout_p,
            num_layers=self.configs.model.num_layers,
        )

    def collect_outputs(
            self,
            stage: str,
            logits: torch.Tensor,
            targets: torch.Tensor,
    ) -> OrderedDict:
        perplexity = self.criterion(logits, targets[:, 1:])
        predictions = logits.max(-1)[1]

        self.info({
            f"{stage}_perplexity": perplexity,
            "learning_rate": self.get_lr(),
        })

        return OrderedDict({
            "loss": perplexity,
            "logits": logits,
            "targets": targets,
            "predictions": predictions,
        })

    def forward(self, inputs: torch.Tensor, input_lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.lm(inputs, input_lengths)
        predictions = logits.max(-1)[1]
        return {
            "predictions": predictions,
            "logits": logits,
        }

    def training_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, input_lengths, targets = batch
        logits = self.lm(inputs, input_lengths)
        return self.collect_outputs(
            stage='train',
            logits=logits,
            targets=targets,
        )

    def validation_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, input_lengths, targets = batch
        logits = self.lm(inputs, input_lengths)
        return self.collect_outputs(
            stage='val',
            logits=logits,
            targets=targets,
        )

    def test_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, input_lengths, targets = batch
        logits = self.lm(inputs, input_lengths)
        return self.collect_outputs(
            stage='test',
            logits=logits,
            targets=targets,
        )
