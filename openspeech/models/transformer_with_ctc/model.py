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

from omegaconf import DictConfig
from torch import Tensor
from typing import Dict
from collections import OrderedDict

from openspeech.models import register_model, OpenspeechCTCModel
from openspeech.encoders import TransformerEncoder
from openspeech.models.transformer_with_ctc.configurations import TransformerWithCTCConfigs
from openspeech.modules.wrapper import Linear
from openspeech.vocabs.vocab import Vocabulary


@register_model('transformer_with_ctc', dataclass=TransformerWithCTCConfigs)
class TransformerWithCTCModel(OpenspeechCTCModel):
    r"""
    Transformer Encoder Only Model.

    Args:
        configs (DictConfig): configuration set.
        vocab (Vocabulary): the class of vocabulary

    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
            `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        * dict (dict): Result of model predictions that contains `y_hats`, `logits`, `output_lengths`
    """

    def __init__(self, configs: DictConfig, vocab: Vocabulary, ) -> None:
        super(TransformerWithCTCModel, self).__init__(configs, vocab)
        self.fc = Linear(self.configs.model.d_model, self.num_classes, bias=False)

    def build_model(self):
        self.encoder = TransformerEncoder(
            input_dim=self.configs.audio.num_mels,
            d_model=self.configs.model.d_model,
            d_ff=self.configs.model.d_ff,
            num_layers=self.configs.model.num_encoder_layers,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.encoder_dropout_p,
            joint_ctc_attention=False,
            num_classes=self.num_classes,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Dict[str, Tensor]:
        r"""
        Forward propagate a `inputs` and `targets` pair for inference.

        Inputs:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * dict (dict): Result of model predictions that contains `y_hats`, `logits`, `output_lengths`
        """
        return super(TransformerWithCTCModel, self).forward(inputs, input_lengths)

    def training_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch
        logits, encoder_logits, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(logits).log_softmax(dim=-1)
        return self.collect_outputs(
            stage='train',
            logits=logits,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def validation_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch
        logits, encoder_logits, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(logits).log_softmax(dim=-1)
        return self.collect_outputs(
            stage='valid',
            logits=logits,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def test_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch
        logits, encoder_logits, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(logits).log_softmax(dim=-1)
        return self.collect_outputs(
            stage='test',
            logits=logits,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )
