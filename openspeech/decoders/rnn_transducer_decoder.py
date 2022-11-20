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

from typing import Tuple

import torch
import torch.nn as nn

from openspeech.decoders import OpenspeechDecoder
from openspeech.modules import Linear


class RNNTransducerDecoder(OpenspeechDecoder):
    r"""
    Decoder of RNN-Transducer

    Args:
        num_classes (int): number of classification
        hidden_state_dim (int, optional): hidden state dimension of decoders (default: 512)
        output_dim (int, optional): output dimension of encoders and decoders (default: 512)
        num_layers (int, optional): number of decoders layers (default: 1)
        rnn_type (str, optional): type of rnn cell (default: lstm)
        sos_id (int, optional): start of sentence identification
        eos_id (int, optional): end of sentence identification
        dropout_p (float, optional): dropout probability of decoders

    Inputs: inputs, input_lengths
        inputs (torch.LongTensor): A target sequence passed to decoders. `IntTensor` of size ``(batch, seq_length)``
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        hidden_states (torch.FloatTensor): A previous hidden state of decoders. `FloatTensor` of size ``(batch, seq_length, dimension)``

    Returns:
        (Tensor, Tensor):

        * decoder_outputs (torch.FloatTensor): A output sequence of decoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        * hidden_states (torch.FloatTensor): A hidden state of decoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``

    Reference:
        A Graves: Sequence Transduction with Recurrent Neural Networks
        https://arxiv.org/abs/1211.3711.pdf
    """
    supported_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
        self,
        num_classes: int,
        hidden_state_dim: int,
        output_dim: int,
        num_layers: int,
        rnn_type: str = "lstm",
        pad_id: int = 0,
        sos_id: int = 1,
        eos_id: int = 2,
        dropout_p: float = 0.2,
    ):
        super(RNNTransducerDecoder, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.pad_id = (pad_id,)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.embedding = nn.Embedding(num_classes, hidden_state_dim)
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
        self.out_proj = Linear(hidden_state_dim, output_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor = None,
        hidden_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propage a `inputs` (targets) for training.

        Inputs:
            inputs (torch.LongTensor): A input sequence passed to label encoder. Typically inputs will be a padded `LongTensor` of size ``(batch, target_length)``
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
            hidden_states (torch.FloatTensor): Previous hidden states.

        Returns:
            (Tensor, Tensor):

            * outputs (torch.FloatTensor): A output sequence of decoders. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            * hidden_states (torch.FloatTensor): A hidden state of decoders. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
        """
        embedded = self.embedding(inputs)

        if hidden_states is not None:
            outputs, hidden_states = self.rnn(embedded, hidden_states)
        else:
            outputs, hidden_states = self.rnn(embedded)

        outputs = self.out_proj(outputs)
        return outputs, hidden_states
