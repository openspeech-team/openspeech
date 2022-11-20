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

from dataclasses import dataclass, field

from openspeech.dataclass.configurations import OpenspeechDataclass


@dataclass
class RNNTransducerConfigs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.RNNTransducer`.

    It is used to initiated an `RNNTransducer` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: transformer_transducer)
        encoder_hidden_state_dim (int): Hidden state dimension of encoder (default: 312)
        decoder_hidden_state_dim (int): Hidden state dimension of decoder (default: 512)
        num_encoder_layers (int): The number of encoder layers. (default: 4)
        num_decoder_layers (int): The number of decoder layers. (default: 1)
        encoder_dropout_p (float): The dropout probability of encoder. (default: 0.2)
        decoder_dropout_p (float): The dropout probability of decoder. (default: 0.2)
        bidirectional (bool): If True, becomes a bidirectional encoders (default: True)
        rnn_type (str): Type of rnn cell (rnn, lstm, gru) (default: lstm)
        output_dim (int): dimension of model output. (default: 512)
        optimizer (str): Optimizer for training. (default: adam)
    """
    model_name: str = field(default="rnn_transducer", metadata={"help": "Model name"})
    encoder_hidden_state_dim: int = field(default=320, metadata={"help": "Dimension of encoder."})
    decoder_hidden_state_dim: int = field(default=512, metadata={"help": "Dimension of decoder."})
    num_encoder_layers: int = field(default=4, metadata={"help": "The number of encoder layers."})
    num_decoder_layers: int = field(default=1, metadata={"help": "The number of decoder layers."})
    encoder_dropout_p: float = field(default=0.2, metadata={"help": "The dropout probability of encoder."})
    decoder_dropout_p: float = field(default=0.2, metadata={"help": "The dropout probability of decoder."})
    bidirectional: bool = field(default=True, metadata={"help": "If True, becomes a bidirectional encoders"})
    rnn_type: str = field(default="lstm", metadata={"help": "Type of rnn cell (rnn, lstm, gru)"})
    output_dim: int = field(default=512, metadata={"help": "Dimension of outputs"})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})
