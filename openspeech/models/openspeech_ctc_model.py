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

import torch
from omegaconf import DictConfig

from openspeech.models import OpenspeechModel
from openspeech.tokenizers.tokenizer import Tokenizer


class OpenspeechCTCModel(OpenspeechModel):
    r"""
    Base class for OpenSpeech's encoder-only models (ctc-model).

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        ouputs (dict): Result of model predictions that contains `y_hats`, `logits`, `output_lengths`
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(OpenspeechCTCModel, self).__init__(configs, tokenizer)
        self.encoder = None
        self.decoder = None

    def set_beam_decoder(self, beam_size: int = 3):
        """Setting beam search decoder"""
        from openspeech.search import BeamSearchCTC

        self.decoder = BeamSearchCTC(
            labels=self.tokenizer.labels,
            blank_id=self.tokenizer.blank_id,
            beam_size=beam_size,
        )

    def collect_outputs(
        self,
        stage: str,
        logits: torch.FloatTensor,
        output_lengths: torch.IntTensor,
        targets: torch.IntTensor,
        target_lengths: torch.IntTensor,
    ) -> OrderedDict:
        loss = self.criterion(
            log_probs=logits.transpose(0, 1),
            targets=targets[:, 1:],
            input_lengths=output_lengths,
            target_lengths=target_lengths,
        )
        predictions = logits.max(-1)[1]

        wer = self.wer_metric(targets[:, 1:], predictions)
        cer = self.cer_metric(targets[:, 1:], predictions)

        self.info(
            {
                f"{stage}_wer": wer,
                f"{stage}_cer": cer,
                f"{stage}_loss": loss,
                "learning_rate": self.get_lr(),
            }
        )

        return OrderedDict(
            {
                "loss": loss,
                "wer": wer,
                "cer": cer,
            }
        )

    def forward(self, inputs: torch.FloatTensor, input_lengths: torch.IntTensor) -> Dict[str, torch.Tensor]:
        r"""
        Forward propagate a `inputs` and `targets` pair for inference.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.IntTensor): The length of input tensor. ``(batch)``

        Returns:
            ouputs (dict): Result of model predictions that contains `y_hats`, `logits`, `output_lengths`
        """
        outputs = self.encoder(inputs, input_lengths)

        if len(outputs) == 2:
            logits, output_lengths = outputs
        else:
            logits, _, output_lengths = outputs

        logits = self.fc(logits).log_softmax(dim=-1)

        if self.decoder is not None:
            y_hats = self.decoder(logits)
        else:
            y_hats = logits.max(-1)[1]
        return {
            "predictions": y_hats,
            "logits": logits,
            "output_lengths": output_lengths,
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
        logits, output_lengths = self.encoder(inputs, input_lengths)
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
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        inputs, targets, input_lengths, target_lengths = batch
        logits, output_lengths = self.encoder(inputs, input_lengths)
        return self.collect_outputs(
            stage="val",
            logits=logits,
            output_lengths=output_lengths,
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
        logits, output_lengths = self.encoder(inputs, input_lengths)
        return self.collect_outputs(
            stage="test",
            logits=logits,
            output_lengths=output_lengths,
            targets=targets,
            target_lengths=target_lengths,
        )
