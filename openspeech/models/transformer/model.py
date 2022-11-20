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

from omegaconf import DictConfig

from openspeech.decoders import TransformerDecoder
from openspeech.encoders import ConvolutionalTransformerEncoder, TransformerEncoder
from openspeech.models import OpenspeechCTCModel, OpenspeechEncoderDecoderModel, register_model
from openspeech.models.transformer.configurations import (
    JointCTCTransformerConfigs,
    TransformerConfigs,
    TransformerWithCTCConfigs,
    VGGTransformerConfigs,
)
from openspeech.modules import Linear
from openspeech.tokenizers.tokenizer import Tokenizer


@register_model("transformer", dataclass=TransformerConfigs)
class TransformerModel(OpenspeechEncoderDecoderModel):
    r"""
    A Speech Transformer model. User is able to modify the attributes as needed.
    The model is based on the paper "Attention Is All You Need".

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokeizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions.
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(TransformerModel, self).__init__(configs, tokenizer)

        self.encoder = TransformerEncoder(
            input_dim=self.configs.audio.num_mels,
            d_model=self.configs.model.d_model,
            d_ff=self.configs.model.d_ff,
            num_layers=self.configs.model.num_encoder_layers,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.encoder_dropout_p,
            joint_ctc_attention=self.configs.model.joint_ctc_attention,
            num_classes=self.num_classes,
        )
        self.decoder = TransformerDecoder(
            num_classes=self.num_classes,
            d_model=self.configs.model.d_model,
            d_ff=self.configs.model.d_ff,
            num_layers=self.configs.model.num_decoder_layers,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.decoder_dropout_p,
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            max_length=self.configs.model.max_length,
        )

    def set_beam_decoder(self, beam_size: int = 3):
        """Setting beam search decoder"""
        from openspeech.search import BeamSearchTransformer

        self.decoder = BeamSearchTransformer(
            decoder=self.decoder,
            beam_size=beam_size,
        )


@register_model("joint_ctc_transformer", dataclass=JointCTCTransformerConfigs)
class JointCTCTransformerModel(OpenspeechEncoderDecoderModel):
    r"""
    A Speech Transformer model. User is able to modify the attributes as needed.
    The model is based on the paper "Attention Is All You Need".

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions.
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(JointCTCTransformerModel, self).__init__(configs, tokenizer)

        self.encoder = ConvolutionalTransformerEncoder(
            input_dim=self.configs.audio.num_mels,
            extractor=self.configs.extractor,
            d_model=self.configs.model.d_model,
            d_ff=self.configs.model.d_ff,
            num_layers=self.configs.model.num_encoder_layers,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.encoder_dropout_p,
            joint_ctc_attention=self.configs.model.joint_ctc_attention,
            num_classes=self.num_classes,
        )
        self.decoder = TransformerDecoder(
            num_classes=self.num_classes,
            d_model=self.configs.model.d_model,
            d_ff=self.configs.model.d_ff,
            num_layers=self.configs.model.num_decoder_layers,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.decoder_dropout_p,
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            max_length=self.configs.model.max_length,
        )

    def set_beam_decoder(self, beam_size: int = 3, n_best: int = 1):
        """Setting beam search decoder"""
        from openspeech.search import BeamSearchTransformer

        self.decoder = BeamSearchTransformer(
            decoder=self.decoder,
            beam_size=beam_size,
        )


@register_model("transformer_with_ctc", dataclass=TransformerWithCTCConfigs)
class TransformerWithCTCModel(OpenspeechCTCModel):
    r"""
    Transformer Encoder Only Model.

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
        super(TransformerWithCTCModel, self).__init__(configs, tokenizer)
        self.fc = Linear(self.configs.model.d_model, self.num_classes, bias=False)

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
        logits, encoder_logits, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(logits).log_softmax(dim=-1)
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
        logits, encoder_logits, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(logits).log_softmax(dim=-1)
        return self.collect_outputs(
            stage="test",
            logits=logits,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )


@register_model("vgg_transformer", dataclass=VGGTransformerConfigs)
class VGGTransformerModel(OpenspeechEncoderDecoderModel):
    r"""
    A Speech Transformer model. User is able to modify the attributes as needed.
    The model is based on the paper "Attention Is All You Need".

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions.
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(VGGTransformerModel, self).__init__(configs, tokenizer)

        self.encoder = ConvolutionalTransformerEncoder(
            input_dim=self.configs.audio.num_mels,
            extractor=self.configs.extractor,
            d_model=self.configs.model.d_model,
            d_ff=self.configs.model.d_ff,
            num_layers=self.configs.model.num_encoder_layers,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.encoder_dropout_p,
            joint_ctc_attention=self.configs.model.joint_ctc_attention,
            num_classes=self.num_classes,
        )
        self.decoder = TransformerDecoder(
            num_classes=self.num_classes,
            d_model=self.configs.model.d_model,
            d_ff=self.configs.model.d_ff,
            num_layers=self.configs.model.num_decoder_layers,
            num_heads=self.configs.model.num_attention_heads,
            dropout_p=self.configs.model.decoder_dropout_p,
            pad_id=self.tokenizer.pad_id,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            max_length=self.configs.model.max_length,
        )

    def set_beam_decoder(self, beam_size: int = 3):
        """Setting beam search decoder"""
        from openspeech.search import BeamSearchTransformer

        self.decoder = BeamSearchTransformer(
            decoder=self.decoder,
            beam_size=beam_size,
        )
