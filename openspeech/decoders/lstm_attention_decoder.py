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

import random
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from openspeech.decoders import OpenspeechDecoder
from openspeech.modules import (
    AdditiveAttention,
    DotProductAttention,
    Linear,
    LocationAwareAttention,
    MultiHeadAttention,
    View,
)


class LSTMAttentionDecoder(OpenspeechDecoder):
    r"""
    Converts higher level features (from encoders) into output utterances
    by specifying a probability distribution over sequences of characters.

    Args:
        num_classes (int): number of classification
        hidden_state_dim (int): the number of features in the decoders hidden state `h`
        num_layers (int, optional): number of recurrent layers (default: 2)
        rnn_type (str, optional): type of RNN cell (default: lstm)
        pad_id (int, optional): index of the pad symbol (default: 0)
        sos_id (int, optional): index of the start of sentence symbol (default: 1)
        eos_id (int, optional): index of the end of sentence symbol (default: 2)
        attn_mechanism (str, optional): type of attention mechanism (default: multi-head)
        num_heads (int, optional): number of attention heads. (default: 4)
        dropout_p (float, optional): dropout probability of decoders (default: 0.2)

    Inputs: inputs, encoder_outputs, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_state_dim): tensor with containing the outputs of the encoders.
          Used for attention mechanism (default is `None`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Returns: logits
        * logits (torch.FloatTensor) : log probabilities of model's prediction
    """
    supported_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
        self,
        num_classes: int,
        max_length: int = 150,
        hidden_state_dim: int = 1024,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        attn_mechanism: str = "multi-head",
        num_heads: int = 4,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout_p: float = 0.3,
    ) -> None:
        super(LSTMAttentionDecoder, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.attn_mechanism = attn_mechanism.lower()
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
        self.input_dropout = nn.Dropout(dropout_p)
        rnn_cell = self.supported_rnns[rnn_type.lower()]
        self.rnn = rnn_cell(
            input_size=hidden_state_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=False,
        )

        if self.attn_mechanism == "loc":
            self.attention = LocationAwareAttention(hidden_state_dim, attn_dim=hidden_state_dim, smoothing=False)
        elif self.attn_mechanism == "multi-head":
            self.attention = MultiHeadAttention(hidden_state_dim, num_heads=num_heads)
        elif self.attn_mechanism == "additive":
            self.attention = AdditiveAttention(hidden_state_dim)
        elif self.attn_mechanism == "dot":
            self.attention = DotProductAttention(dim=hidden_state_dim)
        elif self.attn_mechanism == "scaled-dot":
            self.attention = DotProductAttention(dim=hidden_state_dim, scale=True)
        else:
            raise ValueError("Unsupported attention: %s".format(attn_mechanism))

        self.fc = nn.Sequential(
            Linear(hidden_state_dim << 1, hidden_state_dim),
            nn.Tanh(),
            View(shape=(-1, self.hidden_state_dim), contiguous=True),
            Linear(hidden_state_dim, num_classes),
        )

    def forward_step(
        self,
        input_var: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        encoder_outputs: torch.Tensor,
        attn: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, output_lengths = input_var.size(0), input_var.size(1)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        outputs, hidden_states = self.rnn(embedded, hidden_states)

        if self.attn_mechanism == "loc":
            context, attn = self.attention(outputs, encoder_outputs, attn)
        else:
            context, attn = self.attention(outputs, encoder_outputs, encoder_outputs)

        outputs = torch.cat((outputs, context), dim=2)

        step_outputs = self.fc(outputs.view(-1, self.hidden_state_dim << 1)).log_softmax(dim=-1)
        step_outputs = step_outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return step_outputs, hidden_states, attn

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        encoder_output_lengths: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward propagate a `encoder_outputs` for training.

        Args:
            targets (torch.LongTensr): A target sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            encoder_output_lengths: The length of encoders outputs. ``(batch)``
            teacher_forcing_ratio (float): ratio of teacher forcing

        Returns:
            * logits (torch.FloatTensor): Log probability of model predictions.
        """
        logits = list()
        hidden_states, attn = None, None

        targets, batch_size, max_length = self.validate_args(targets, encoder_outputs, teacher_forcing_ratio)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            targets = targets[targets != self.eos_id].view(batch_size, -1)

            if self.attn_mechanism == "loc" or self.attn_mechanism == "additive":
                for di in range(targets.size(1)):
                    input_var = targets[:, di].unsqueeze(1)
                    step_outputs, hidden_states, attn = self.forward_step(
                        input_var=input_var,
                        hidden_states=hidden_states,
                        encoder_outputs=encoder_outputs,
                        attn=attn,
                    )
                    logits.append(step_outputs)

            else:
                step_outputs, hidden_states, attn = self.forward_step(
                    input_var=targets,
                    hidden_states=hidden_states,
                    encoder_outputs=encoder_outputs,
                    attn=attn,
                )

                for di in range(step_outputs.size(1)):
                    step_output = step_outputs[:, di, :]
                    logits.append(step_output)

        else:
            input_var = targets[:, 0].unsqueeze(1)

            for di in range(max_length):
                step_outputs, hidden_states, attn = self.forward_step(
                    input_var=input_var,
                    hidden_states=hidden_states,
                    encoder_outputs=encoder_outputs,
                    attn=attn,
                )
                logits.append(step_outputs)
                input_var = logits[-1].topk(1)[1]

        logits = torch.stack(logits, dim=1)

        return logits

    def validate_args(
        self,
        targets: Optional[Any] = None,
        encoder_outputs: torch.Tensor = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> Tuple[torch.Tensor, int, int]:
        assert encoder_outputs is not None
        batch_size = encoder_outputs.size(0)

        if targets is None:  # inference
            targets = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            max_length = self.max_length

            if torch.cuda.is_available():
                targets = targets.cuda()

            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no targets is provided.")

        else:
            max_length = targets.size(1) - 1  # minus the start of sequence symbol

        return targets, batch_size, max_length
