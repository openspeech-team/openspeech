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
import torch.nn as nn
import warnings
from torch import Tensor
from collections import OrderedDict
from omegaconf import DictConfig
from typing import Tuple, Dict

from openspeech.models import OpenspeechModel
from openspeech.modules import Linear
from openspeech.utils import get_class_name
from openspeech.vocabs.vocab import Vocabulary


class OpenspeechTransducerModel(OpenspeechModel):
    r"""
    Base class for OpenSpeech's transducer models.

    Args:
        configs (DictConfig): configuration set.
        vocab (Vocabulary): the class of vocabulary

    Inputs:
        - **inputs** (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be
            a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        - **input_lengths** (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        - **y_hats** (torch.FloatTensor): Result of model predictions.
    """

    def __init__(self, configs: DictConfig, vocab: Vocabulary, ) -> None:
        super(OpenspeechTransducerModel, self).__init__(configs, vocab)
        self.encoder = None
        self.decoder = None

        if hasattr(self.configs.model, "encoder_dim"):
            in_features = self.configs.model.encoder_dim + self.configs.model.decoder_output_dim
        elif hasattr(self.configs.model, "output_dim"):
            in_features = self.configs.model.output_dim << 1
        else:
            raise ValueError("Transducer model must be contain `encoder_dim` or `encoder_hidden_state_dim` config.")

        self.fc = nn.Sequential(
            Linear(in_features=in_features, out_features=in_features),
            nn.Tanh(),
            Linear(in_features=in_features, out_features=self.num_classes),
        )

    def set_beam_decoder(self, beam_size: int = 3):
        warnings.warn("Currently, Beamsearch has not yet been implemented in the transducer model.")

    def collect_outputs(
            self,
            stage: str,
            logits: torch.FloatTensor,
            input_lengths: torch.IntTensor,
            targets: torch.IntTensor,
            target_lengths: torch.IntTensor,
            predictions: torch.Tensor = None,
    ) -> OrderedDict:
        if predictions is None:
            predictions = logits.max(-1)[1]
            loss = self.criterion(
                logits=logits,
                targets=targets[:, 1:].contiguous().int(),
                input_lengths=input_lengths.int(),
                target_lengths=target_lengths.int(),
            )

            wer = self.wer_metric(targets[:, 1:], predictions)
            cer = self.cer_metric(targets[:, 1:], predictions)

            self.log_steps(stage, wer, cer, loss)

            tqdm_dict = {
                f"{stage}_loss": loss,
                "wer": wer,
                "cer": cer,
            }

            return OrderedDict({
                "loss": loss,
                "progress_bar": tqdm_dict,
                "log": tqdm_dict,
            })

        else:
            wer = self.wer_metric(targets[:, 1:], predictions)
            cer = self.cer_metric(targets[:, 1:], predictions)
            self.log_steps(stage, wer, cer)

            progress_bar_dict = {
                f"{stage}_loss": None,
                "wer": wer,
                "cer": cer,
            }

            return OrderedDict({
                "loss": None,
                "progress_bar": progress_bar_dict,
                "log": progress_bar_dict,
            })

    def _expand_for_joint(self, encoder_outputs: Tensor, decoder_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        input_length = encoder_outputs.size(1)
        target_length = decoder_outputs.size(1)

        encoder_outputs = encoder_outputs.unsqueeze(2)
        decoder_outputs = decoder_outputs.unsqueeze(1)

        encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
        decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])
        return encoder_outputs, decoder_outputs

    def joint(self, encoder_outputs: Tensor, decoder_outputs: Tensor) -> Tensor:
        r"""
        Joint `encoder_outputs` and `decoder_outputs`.

        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            decoder_outputs (torch.FloatTensor): A output sequence of decoders. `FloatTensor` of size
                ``(batch, seq_length, dimension)``

        Returns:
            * outputs (torch.FloatTensor): outputs of joint `encoder_outputs` and `decoder_outputs`..
        """
        if encoder_outputs.dim() == 3 and decoder_outputs.dim() == 3:
            encoder_outputs, decoder_outputs = self._expand_for_joint(encoder_outputs, decoder_outputs)
        else:
            assert encoder_outputs.dim() == decoder_outputs.dim()

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)
        outputs = self.fc(outputs).log_softmax(dim=-1)

        return outputs

    def decode(self, encoder_output: Tensor, max_length: int) -> Tensor:
        r"""
        Decode `encoder_outputs`.

        Args:
            encoder_output (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
                ``(seq_length, dimension)``
            max_length (int): max decoding time step

        Returns:
            * logits (torch.FloatTensor): Log probability of model predictions.
        """
        pred_tokens = list()
        decoder_input = encoder_output.new_zeros(1, 1).fill_(self.decoder.sos_id).long()
        decoder_output, hidden_state = self.decoder(decoder_input)

        for t in range(max_length):
            step_output = self.joint(encoder_output[t].view(-1), decoder_output.view(-1))

            pred_token = step_output.argmax(dim=0)
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)

            decoder_input = torch.LongTensor([[pred_token]])
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()

            decoder_output, hidden_state = self.decoder(
                decoder_input, hidden_states=hidden_state
            )

        return torch.LongTensor(pred_tokens)

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Dict[str, Tensor]:
        r"""
        Decode `encoder_outputs`.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            * dict (dict): Result of model predictions that contains `predictions`, `logits`,
                `encoder_outputs`, `encoder_output_lengths`
        """
        outputs = list()

        if get_class_name(self.encoder) == "TransducerEncoderBase":
            encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        else:
            encoder_outputs, _, output_lengths = self.encoder(inputs, input_lengths)
        max_length = encoder_outputs.size(1)

        for encoder_output in encoder_outputs:
            decoded_seq = self.decode(encoder_output, max_length)
            outputs.append(decoded_seq)

        logits = torch.stack(outputs, dim=1).transpose(0, 1)
        predictions = logits.max(-1)[1]
        return {
            "predictions": predictions,
            "logits": logits,
            "encoder_outputs": encoder_outputs,
            "encoder_output_lengths": output_lengths,
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

        if get_class_name(self.encoder) == "TransducerEncoderBase":
            encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        else:
            encoder_outputs, _, output_lengths = self.encoder(inputs, input_lengths)

        decoder_outputs, _ = self.decoder(targets, target_lengths)
        logits = self.joint(encoder_outputs, decoder_outputs)

        return self.collect_outputs(
            'train',
            logits=logits,
            input_lengths=input_lengths,
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
        predictions = list()
        inputs, targets, input_lengths, target_lengths = batch

        if get_class_name(self.encoder) == "TransducerEncoderBase":
            encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        else:
            encoder_outputs, _, output_lengths = self.encoder(inputs, input_lengths)

        max_length = encoder_outputs.size(1)

        for idx, encoder_output in enumerate(encoder_outputs):
            prediction = self.decode(encoder_output, max_length)
            predictions.append(prediction)

        predictions = torch.stack(predictions)

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
            train_batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        predictions = list()
        inputs, targets, input_lengths, target_lengths = batch

        if get_class_name(self.encoder) == "TransducerEncoderBase":
            encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        else:
            encoder_outputs, _, output_lengths = self.encoder(inputs, input_lengths)

        max_length = encoder_outputs.size(1)

        for idx, encoder_output in enumerate(encoder_outputs):
            prediction = self.decode(encoder_output, max_length)
            predictions.append(prediction)

        predictions = torch.stack(predictions)

        return self.collect_outputs(
            'test',
            logits=None,
            input_lengths=input_lengths,
            targets=targets,
            target_lengths=target_lengths,
            predictions=predictions,
        )
