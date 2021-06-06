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

from openspeech.models import register_model, OpenspeechTransducerModel
from openspeech.models.conformer.model import ConformerEncoder
from openspeech.models.conformer_transducer.configurations import ConformerTransducerConfigs
from openspeech.models.rnn_transducer.model import RNNTransducerDecoder
from openspeech.vocabs.vocab import Vocabulary


@register_model('conformer_transducer', dataclass=ConformerTransducerConfigs)
class ConformerTransducerModel(OpenspeechTransducerModel):
    r"""
    Conformer: Convolution-augmented Transformer for Speech Recognition
    Paper: https://arxiv.org/abs/2005.08100

    Args:
        configs (DictConfig): configuraion set
        vocab (Vocabulary): vocab of training data

    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
            `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        * outputs (dict): Result of model predictions.
    """

    def __init__(self, configs: DictConfig, vocab: Vocabulary, ) -> None:
        super(ConformerTransducerModel, self).__init__(configs, vocab)

    def build_model(self):
        self.encoder = ConformerEncoder(
            num_classes=self.num_classes,
            input_dim=self.configs.audio.num_mels,
            encoder_dim=self.configs.model.encoder_dim,
            num_layers=self.configs.model.num_encoder_layers,
            num_attention_heads=self.configs.model.num_attention_heads,
            feed_forward_expansion_factor=self.configs.model.feed_forward_expansion_factor,
            conv_expansion_factor=self.configs.model.conv_expansion_factor,
            input_dropout_p=self.configs.model.input_dropout_p,
            feed_forward_dropout_p=self.configs.model.feed_forward_dropout_p,
            attention_dropout_p=self.configs.model.attention_dropout_p,
            conv_dropout_p=self.configs.model.conv_dropout_p,
            conv_kernel_size=self.configs.model.conv_kernel_size,
            half_step_residual=self.configs.model.half_step_residual,
            joint_ctc_attention=False,
        )
        self.decoder = RNNTransducerDecoder(
            num_classes=self.num_classes,
            hidden_state_dim=self.configs.model.decoder_hidden_state_dim,
            output_dim=self.configs.model.decoder_output_dim,
            num_layers=self.configs.model.num_decoder_layers,
            rnn_type=self.configs.model.rnn_type,
            sos_id=self.vocab.sos_id,
            eos_id=self.vocab.eos_id,
            dropout_p=self.configs.model.decoder_dropout_p,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Dict[str, Tensor]:
        r"""
        Forward propagate a `inputs` and `targets` pair for inference.

        Inputs:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * outputs (dict): Result of model predictions.
        """
        return super(ConformerTransducerModel, self).forward(inputs, input_lengths)

    def training_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        return super(ConformerTransducerModel, self).training_step(batch, batch_idx)

    def validation_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        return super(ConformerTransducerModel, self).validation_step(batch, batch_idx)

    def test_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        return super(ConformerTransducerModel, self).test_step(batch, batch_idx)
