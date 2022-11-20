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
from typing import Optional, Tuple

import torch
import torch.nn as nn

from openspeech.lm.openspeech_lm import OpenspeechLanguageModelBase
from openspeech.modules import Linear, View


class LSTMForLanguageModel(OpenspeechLanguageModelBase):
    """
    Language Modelling is the core problem for a number of of natural language processing tasks such as speech to text,
    conversational system, and text summarization. A trained language model learns the likelihood of occurrence
    of a word based on the previous sequence of words used in the text.

    Args:
        num_classes (int): number of classification
        max_length (int): max decoding length (default: 128)
        hidden_state_dim (int): dimension of hidden state vector (default: 768)
        rnn_type (str, optional): type of RNN cell (default: lstm)
        pad_id (int, optional): index of the pad symbol (default: 0)
        sos_id (int, optional): index of the start of sentence symbol (default: 1)
        eos_id (int, optional): index of the end of sentence symbol (default: 2)
        num_layers (int, optional): number of recurrent layers (default: 2)
        dropout_p (float, optional): dropout probability of decoders (default: 0.2)

    Inputs: inputs, teacher_forcing_ratio
        inputs (torch.LongTensr): A input sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
        teacher_forcing_ratio (float): ratio of teacher forcing

    Returns:
        * logits (torch.FloatTensor): Log probability of model predictions.
    """

    supported_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
        self,
        num_classes: int,
        max_length: int = 128,
        hidden_state_dim: int = 768,
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout_p: float = 0.3,
    ) -> None:
        super(LSTMForLanguageModel, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.max_length = max_length
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.pad_id = pad_id
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

        self.fc = nn.Sequential(
            Linear(hidden_state_dim, hidden_state_dim),
            nn.Tanh(),
            View(shape=(-1, self.hidden_state_dim), contiguous=True),
            Linear(hidden_state_dim, num_classes),
        )

    def forward_step(
        self,
        input_var: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, output_lengths = input_var.size(0), input_var.size(1)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        if self.training:
            self.rnn.flatten_parameters()

        outputs, hidden_states = self.rnn(embedded, hidden_states)

        step_outputs = self.fc(outputs.reshape(-1, self.hidden_state_dim)).log_softmax(dim=-1)
        step_outputs = step_outputs.view(batch_size, output_lengths, -1).squeeze(1)

        return step_outputs, hidden_states

    def forward(
        self,
        inputs: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward propagate a `encoder_outputs` for training.

        Args:
            inputs (torch.LongTensr): A input sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
            teacher_forcing_ratio (float): ratio of teacher forcing

        Returns:
            * logits (torch.FloatTensor): Log probability of model predictions.
        """
        batch_size = inputs.size(0)
        logits, hidden_states = list(), None
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            inputs = inputs[inputs != self.eos_id].view(batch_size, -1)
            step_outputs, hidden_states = self.forward_step(input_var=inputs, hidden_states=hidden_states)

            for di in range(step_outputs.size(1)):
                step_output = step_outputs[:, di, :]
                logits.append(step_output)

        else:
            input_var = inputs[:, 0].unsqueeze(1)
            for di in range(self.max_length):
                step_output, hidden = self.forward_step(input_var=input_var, hidden_states=hidden_states)

                step_output = step_output.squeeze(1)
                logits.append(step_output)
                input_var = logits[-1].topk(1)[1]

        return torch.stack(logits, dim=1)
