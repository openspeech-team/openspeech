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
from torch import Tensor
from typing import Dict
from collections import OrderedDict

from openspeech.models import register_model, OpenspeechTransducerModel
from openspeech.decoders import TransformerTransducerDecoder
from openspeech.encoders import TransformerTransducerEncoder
from openspeech.models.transformer_transducer.configurations import TransformerTransducerConfigs
from openspeech.vocabs.vocab import Vocabulary


@register_model('transformer_transducer', dataclass=TransformerTransducerConfigs)
class TransformerTransducerModel(OpenspeechTransducerModel):
    r"""
    Transformer-Transducer is that every layer is identical for both audio and label encoders.
    Unlike the basic transformer structure, the audio encoder and label encoder are separate.
    So, the alignment is handled by a separate forward-backward process within the RNN-T architecture.
    And we replace the LSTM encoders in RNN-T architecture with Transformer encoders.

    Args:
        configs (DictConfig): configuraion set
        vocab (Vocabulary): vocab of training data

    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
            `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        * y_hats (torch.FloatTensor): Result of model predictions.
    """

    def __init__(self, configs: DictConfig, vocab: Vocabulary, ) -> None:
        super(TransformerTransducerModel, self).__init__(configs, vocab)

    def build_model(self):
        self.encoder = TransformerTransducerEncoder(
            input_size=self.configs.audio.num_mels,
            model_dim=self.configs.model.encoder_dim,
            d_ff=self.configs.model.d_ff,
            num_layers=self.configs.model.num_audio_layers,
            num_heads=self.configs.model.num_attention_heads,
            dropout=self.configs.model.audio_dropout_p,
            max_positional_length=self.configs.model.max_positional_length,
        )
        self.decoder = TransformerTransducerDecoder(
            num_classes=self.num_classes,
            model_dim=self.configs.model.encoder_dim,
            d_ff=self.configs.model.d_ff,
            num_layers=self.configs.model.num_label_layers,
            num_heads=self.configs.model.num_attention_heads,
            dropout=self.configs.model.label_dropout_p,
            max_positional_length=self.configs.model.max_positional_length,
            pad_id=self.vocab.pad_id,
            sos_id=self.vocab.sos_id,
            eos_id=self.vocab.eos_id,
        )

    def decode(self, encoder_outputs: Tensor, max_length: int) -> Tensor:
        r"""
        Decode `encoder_outputs`.

        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
                ``(seq_length, dimension)``
            max_length (int): max decoding time step

        Returns:
            * y_hats (torch.IntTensor): model's predictions.
        """
        batch = encoder_outputs.size(0)
        pred_tokens = list()

        targets = encoder_outputs.new_tensor([self.decoder.sos_id] * batch, dtype=torch.long)

        for i in range(max_length):
            decoder_output, _ = self.decoder(targets, None)
            decoder_output = decoder_output.squeeze(1)
            encoder_output = encoder_outputs[:, i, :]
            targets = self.joint(encoder_output, decoder_output)
            targets = targets.max(1)[1]
            pred_tokens.append(targets)

        pred_tokens = torch.stack(pred_tokens, dim=1)

        return torch.LongTensor(pred_tokens)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Dict[str, Tensor]:
        r"""
        Decode `encoder_outputs`.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * outputs (dict): Result of model predictions.
        """
        encoder_outputs, _ = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        predictions = self.decode(encoder_outputs, max_length)
        return {
            "predictions": predictions,
            "encoder_outputs": encoder_outputs,
        }

    def training_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        return super(TransformerTransducerModel, self).training_step(batch, batch_idx)

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

        encoder_outputs, _ = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        predictions = self.decode(encoder_outputs, max_length)

        return self.collect_outputs(
            'valid',
            logits=None,
            input_lengths=input_lengths,
            targets=targets,
            target_lengths=target_lengths,
            predictions=predictions,
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

        encoder_outputs, _ = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        predictions = self.decode(encoder_outputs, max_length)

        return self.collect_outputs(
            'valid',
            logits=None,
            input_lengths=input_lengths,
            targets=targets,
            target_lengths=target_lengths,
            predictions=predictions,
        )
