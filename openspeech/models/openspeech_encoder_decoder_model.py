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

from openspeech.models import OpenspeechModel
from openspeech.tokenizers.tokenizer import Tokenizer
from openspeech.utils import get_class_name


class OpenspeechEncoderDecoderModel(OpenspeechModel):
    r"""
    Base class for OpenSpeech's encoder-decoder models.

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions that contains `predictions`, `logits`, `encoder_outputs`,
                `encoder_logits`, `encoder_output_lengths`.
    """

    def __init__(
        self,
        configs: DictConfig,
        tokenizer: Tokenizer,
    ) -> None:
        super(OpenspeechEncoderDecoderModel, self).__init__(configs, tokenizer)
        self.teacher_forcing_ratio = configs.model.teacher_forcing_ratio
        self.encoder = None
        self.decoder = None

    def set_beam_decoder(self, beam_size: int = 3):
        raise NotImplementedError

    def collect_outputs(
        self,
        stage: str,
        logits: Tensor,
        encoder_logits: Tensor,
        encoder_output_lengths: Tensor,
        targets: Tensor,
        target_lengths: Tensor,
    ) -> OrderedDict:
        cross_entropy_loss, ctc_loss = None, None

        if get_class_name(self.criterion) == "JointCTCCrossEntropyLoss":
            loss, ctc_loss, cross_entropy_loss = self.criterion(
                encoder_logits=encoder_logits.transpose(0, 1),
                logits=logits,
                output_lengths=encoder_output_lengths,
                targets=targets[:, 1:],
                target_lengths=target_lengths,
            )
            self.info(
                {
                    f"{stage}_loss": loss,
                    f"{stage}_cross_entropy_loss": cross_entropy_loss,
                    f"{stage}_ctc_loss": ctc_loss,
                }
            )
        elif (
            get_class_name(self.criterion) == "LabelSmoothedCrossEntropyLoss"
            or get_class_name(self.criterion) == "CrossEntropyLoss"
        ):
            loss = self.criterion(logits, targets[:, 1:])
            self.info({f"{stage}_loss": loss})
        else:
            raise ValueError(f"Unsupported criterion: {self.criterion}")

        predictions = logits.max(-1)[1]

        wer = self.wer_metric(targets[:, 1:], predictions)
        cer = self.cer_metric(targets[:, 1:], predictions)

        self.info(
            {
                f"{stage}_wer": wer,
                f"{stage}_cer": cer,
            }
        )

        return OrderedDict(
            {
                "loss": loss,
                "cross_entropy_loss": cross_entropy_loss,
                "ctc_loss": ctc_loss,
                "predictions": predictions,
                "targets": targets,
                "logits": logits,
                "learning_rate": self.get_lr(),
            }
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Dict[str, Tensor]:
        r"""
        Forward propagate a `inputs` and `targets` pair for inference.

        Inputs:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            outputs (dict): Result of model predictions that contains `predictions`, `logits`, `encoder_outputs`,
                `encoder_logits`, `encoder_output_lengths`.
        """
        logits = None
        encoder_outputs, encoder_logits, encoder_output_lengths = self.encoder(inputs, input_lengths)

        if get_class_name(self.decoder) in ("BeamSearchLSTM", "BeamSearchTransformer"):
            predictions = self.decoder(encoder_outputs, encoder_output_lengths)
        else:
            logits = self.decoder(
                encoder_outputs=encoder_outputs,
                encoder_output_lengths=encoder_output_lengths,
                teacher_forcing_ratio=0.0,
            )
            predictions = logits.max(-1)[1]
        return {
            "predictions": predictions,
            "logits": logits,
            "encoder_outputs": encoder_outputs,
            "encoder_logits": encoder_logits,
            "encoder_output_lengths": encoder_output_lengths,
        }

    def training_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch

        encoder_outputs, encoder_logits, encoder_output_lengths = self.encoder(inputs, input_lengths)
        if get_class_name(self.decoder) == "TransformerDecoder":
            logits = self.decoder(
                encoder_outputs=encoder_outputs,
                targets=targets,
                encoder_output_lengths=encoder_output_lengths,
                target_lengths=target_lengths,
                teacher_forcing_ratio=self.teacher_forcing_ratio,
            )
        else:
            logits = self.decoder(
                encoder_outputs=encoder_outputs,
                targets=targets,
                encoder_output_lengths=encoder_output_lengths,
                teacher_forcing_ratio=self.teacher_forcing_ratio,
            )

        return self.collect_outputs(
            stage="train",
            logits=logits,
            encoder_logits=encoder_logits,
            encoder_output_lengths=encoder_output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def validation_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch

        encoder_outputs, encoder_logits, encoder_output_lengths = self.encoder(inputs, input_lengths)
        logits = self.decoder(
            encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            teacher_forcing_ratio=0.0,
        )
        return self.collect_outputs(
            stage="val",
            logits=logits,
            encoder_logits=encoder_logits,
            encoder_output_lengths=encoder_output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )

    def test_step(self, batch: tuple, batch_idx: int) -> OrderedDict:
        r"""
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch

        encoder_outputs, encoder_logits, encoder_output_lengths = self.encoder(inputs, input_lengths)
        logits = self.decoder(
            encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            teacher_forcing_ratio=0.0,
        )
        return self.collect_outputs(
            stage="test",
            logits=logits,
            encoder_logits=encoder_logits,
            encoder_output_lengths=encoder_output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )
