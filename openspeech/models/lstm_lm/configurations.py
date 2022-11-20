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
class LSTMLanguageModelConfigs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.LSTMLanguageModel`.

    It is used to initiated an `LSTMLanguageModel` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: lstm_lm)
        num_layers (int): The number of lstm layers. (default: 3)
        hidden_state_dim (int): The hidden state dimension of model. (default: 512)
        dropout_p (float): The dropout probability of encoder. (default: 0.3)
        rnn_type (str): Type of rnn cell (rnn, lstm, gru) (default: lstm)
        max_length (int): Max decoding length. (default: 128)
        teacher_forcing_ratio (float): The ratio of teacher forcing. (default: 1.0)
        optimizer (str): Optimizer for training. (default: adam)
    """
    model_name: str = field(default="lstm_lm", metadata={"help": "Model name"})
    num_layers: int = field(default=3, metadata={"help": "The number of encoder layers."})
    hidden_state_dim: int = field(default=512, metadata={"help": "The hidden state dimension of encoder."})
    dropout_p: float = field(default=0.3, metadata={"help": "The dropout probability of encoder."})
    rnn_type: str = field(default="lstm", metadata={"help": "Type of rnn cell (rnn, lstm, gru)"})
    max_length: int = field(default=128, metadata={"help": "Max decoding length."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": "The ratio of teacher forcing. "})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})
