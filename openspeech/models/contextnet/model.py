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

from collections import OrderedDict
from typing import Dict

from omegaconf import DictConfig
from torch import Tensor

from openspeech.decoders import LSTMAttentionDecoder, RNNTransducerDecoder
from openspeech.encoders import ContextNetEncoder
from openspeech.models import (
    OpenspeechCTCModel,
    OpenspeechEncoderDecoderModel,
    OpenspeechTransducerModel,
    register_model,
)
from openspeech.models.contextnet.configurations import (
    ContextNetConfigs,
    ContextNetLSTMConfigs,
    ContextNetTransducerConfigs,
)
from openspeech.modules.wrapper import Linear
from openspeech.tokenizers.tokenizer import Tokenizer


@register_model("contextnet", dataclass=ContextNetConfigs)
class ContextNetModel(OpenspeechCTCModel):
    r"""
    Conformer Encoder Only Model.

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions that contains `y_hats`, `logits`, `output_lengths`
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ContextNetModel, self).__init__(configs, tokenizer)
        supported_models = {
            "small": 0.5,
            "medium": 1,
            "large": 2,
        }
        alpha = supported_models[self.configs.model.model_size]
        self.fc = Linear(int(self.configs.model.encoder_dim * alpha), self.num_classes, bias=False)

        self.encoder = ContextNetEncoder(
            num_classes=self.num_classes,
            model_size=self.configs.model.model_size,
            input_dim=self.configs.audio.num_mels,
            num_layers=self.configs.model.num_encoder_layers,
            kernel_size=self.configs.model.kernel_size,
            num_channels=self.configs.model.num_channels,
            output_dim=self.configs.model.encoder_dim,
            joint_ctc_attention=False,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Dict[str, Tensor]:
        r"""
        Forward propagate a `inputs` and `targets` pair for inference.

        Inputs:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            outputs (dict): Result of model predictions that contains `y_hats`, `logits`, `output_lengths`
        """
        return super(ContextNetModel, self).forward(inputs, input_lengths)

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
        encoder_outputs, encoder_logits, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(encoder_outputs).log_softmax(dim=-1)
        return self.collect_outputs(
            stage="train",
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
        encoder_outputs, encoder_logits, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(encoder_outputs).log_softmax(dim=-1)
        return self.collect_outputs(
            stage="valid",
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
        encoder_outputs, encoder_logits, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(encoder_outputs).log_softmax(dim=-1)
        return self.collect_outputs(
            stage="test",
            logits=logits,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )


@register_model("contextnet_lstm", dataclass=ContextNetLSTMConfigs)
class ContextNetLSTMModel(OpenspeechEncoderDecoderModel):
    r"""
    ContextNet encoder + LSTM decoder.

    Args:
        configs (DictConfig): configuraion set
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions that contains `y_hats`, `logits`,
            `encoder_outputs`, `encoder_logits`, `encoder_output_lengths`.
    """

    def __init__(
        self,
        configs: DictConfig,
        tokenizer: Tokenizer,
    ) -> None:
        super(ContextNetLSTMModel, self).__init__(configs, tokenizer)

        self.encoder = ContextNetEncoder(
            num_classes=self.num_classes,
            model_size=self.configs.model.model_size,
            input_dim=self.configs.audio.num_mels,
            num_layers=self.configs.model.num_encoder_layers,
            kernel_size=self.configs.model.kernel_size,
            num_channels=self.configs.model.num_channels,
            output_dim=self.configs.model.encoder_dim,
            joint_ctc_attention=False,
        )
        self.decoder = LSTMAttentionDecoder(
            num_classes=self.num_classes,
            max_length=self.configs.model.max_length,
            hidden_state_dim=self.configs.model.encoder_dim,
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.decoder_dropout_p,
            num_layers=self.configs.model.num_decoder_layers,
            attn_mechanism=self.configs.model.decoder_attn_mechanism,
            rnn_type=self.configs.model.rnn_type,
        )

    def set_beam_decoder(self, beam_size: int = 3):
        """Setting beam search decoder"""
        from openspeech.search import BeamSearchLSTM

        self.decoder = BeamSearchLSTM(
            decoder=self.decoder,
            beam_size=beam_size,
        )


@register_model("contextnet_transducer", dataclass=ContextNetTransducerConfigs)
class ContextNetTransducerModel(OpenspeechTransducerModel):
    r"""
    ContextNet: Improving Convolutional Neural Networks for Automatic Speech Recognition with Global Context
    Paper: https://arxiv.org/abs/2005.03191

    Args:
        configs (DictConfig): configuraion set.
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions.
    """

    def __init__(
        self,
        configs: DictConfig,
        tokenizer: Tokenizer,
    ) -> None:
        super(ContextNetTransducerModel, self).__init__(configs, tokenizer)

        self.encoder = ContextNetEncoder(
            num_classes=self.num_classes,
            model_size=self.configs.model.model_size,
            input_dim=self.configs.audio.num_mels,
            num_layers=self.configs.model.num_encoder_layers,
            kernel_size=self.configs.model.kernel_size,
            num_channels=self.configs.model.num_channels,
            output_dim=self.configs.model.encoder_dim,
            joint_ctc_attention=False,
        )
        self.decoder = RNNTransducerDecoder(
            num_classes=self.num_classes,
            hidden_state_dim=self.configs.model.decoder_hidden_state_dim,
            output_dim=self.configs.model.decoder_output_dim,
            num_layers=self.configs.model.num_decoder_layers,
            rnn_type=self.configs.model.rnn_type,
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            dropout_p=self.configs.model.decoder_dropout_p,
        )
