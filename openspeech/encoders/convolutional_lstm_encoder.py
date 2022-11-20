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

from typing import Optional, Tuple

import torch
import torch.nn as nn

from openspeech.encoders import OpenspeechEncoder
from openspeech.modules import Linear, Transpose


class ConvolutionalLSTMEncoder(OpenspeechEncoder):
    r"""
    Converts low level speech signals into higher level features with convolutional extractor.

    Args:
        input_dim (int): dimension of input vector
        num_classes (int): number of classification
        hidden_state_dim (int): the number of features in the encoders hidden state `h`
        num_layers (int, optional): number of recurrent layers (default: 3)
        bidirectional (bool, optional): if True, becomes a bidirectional encoders (default: False)
        extractor (str): type of CNN extractor (default: vgg)
        conv_activation (str): activation function of convolutional extractor (default: hardtanh)
        rnn_type (str, optional): type of RNN cell (default: lstm)
        dropout_p (float, optional): dropout probability of encoders (default: 0.2)
        joint_ctc_attention (bool, optional): flag indication joint ctc attention or not

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths

    Returns: encoder_outputs, encoder_log__probs, output_lengths
        - **encoder_outputs**: tensor containing the encoded features of the input sequence
        - **encoder_log__probs**: tensor containing log probability for encoder_only loss
        - **output_lengths**: list of sequence lengths produced by Listener
    """
    supported_rnns = {
        "lstm": nn.LSTM,
        "gru": nn.GRU,
        "rnn": nn.RNN,
    }

    def __init__(
        self,
        input_dim: int,
        num_classes: int = None,
        hidden_state_dim: int = 512,
        dropout_p: float = 0.3,
        num_layers: int = 3,
        bidirectional: bool = True,
        rnn_type: str = "lstm",
        extractor: str = "vgg",
        conv_activation: str = "hardtanh",
        joint_ctc_attention: bool = False,
    ) -> None:
        super(ConvolutionalLSTMEncoder, self).__init__()
        extractor = self.supported_extractors[extractor.lower()]
        self.conv = extractor(input_dim=input_dim, activation=conv_activation)
        self.conv_output_dim = self.conv.get_output_dim()

        self.num_classes = num_classes
        self.joint_ctc_attention = joint_ctc_attention

        self.hidden_state_dim = hidden_state_dim
        self.rnn = self.supported_rnns[rnn_type.lower()](
            input_size=self.conv_output_dim,
            hidden_size=hidden_state_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=bidirectional,
        )

        if self.joint_ctc_attention:
            self.fc = nn.Sequential(
                Transpose(shape=(1, 2)),
                nn.Dropout(dropout_p),
                Linear(hidden_state_dim << 1, num_classes, bias=False),
            )

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward propagate a `inputs` for  encoders training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor, Tensor):

            * outputs: A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
            * encoder_logits: Log probability of encoders outputs will be passed to CTC Loss.
                If joint_ctc_attention is False, return None.
            * encoder_output_lengths: The length of encoders outputs. ``(batch)``
        """
        encoder_logits = None

        conv_outputs, output_lengths = self.conv(inputs, input_lengths)

        conv_outputs = nn.utils.rnn.pack_padded_sequence(conv_outputs.transpose(0, 1), output_lengths.cpu())
        outputs, hidden_states = self.rnn(conv_outputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs.transpose(0, 1)

        if self.joint_ctc_attention:
            encoder_logits = self.fc(outputs.transpose(1, 2)).log_softmax(dim=2)

        return outputs, encoder_logits, output_lengths
