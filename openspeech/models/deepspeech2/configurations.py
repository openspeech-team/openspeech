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
class DeepSpeech2Configs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.DeepSpeech2`.

    It is used to initiated an `DeepSpeech2` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: deepspeech2)
        num_rnn_layers (int): The number of rnn layers. (default: 5)
        rnn_hidden_dim (int): The hidden state dimension of rnn. (default: 1024)
        dropout_p (float): The dropout probability of model. (default: 0.3)
        bidirectional (bool): If True, becomes a bidirectional encoders (default: True)
        rnn_type (str): Type of rnn cell (rnn, lstm, gru) (default: gru)
        activation (str): Type of activation function (default: str)
        optimizer (str): Optimizer for training. (default: adam)
    """
    model_name: str = field(default="deepspeech2", metadata={"help": "Model name"})
    rnn_type: str = field(default="gru", metadata={"help": "Type of rnn cell (rnn, lstm, gru)"})
    num_rnn_layers: int = field(default=5, metadata={"help": "The number of rnn layers"})
    rnn_hidden_dim: int = field(default=1024, metadata={"help": "Hidden state dimenstion of RNN."})
    dropout_p: float = field(default=0.3, metadata={"help": "The dropout probability of model."})
    bidirectional: bool = field(default=True, metadata={"help": "If True, becomes a bidirectional encoders"})
    activation: str = field(default="hardtanh", metadata={"help": "Type of activation function"})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})
