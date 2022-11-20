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
class ContextNetConfigs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.ContextNet`.

    It is used to initiated an `ContextNet` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: contextnet)
        model_size (str, optional): Size of the model['small', 'medium', 'large'] (default : 'medium')
        input_dim (int, optional): Dimension of input vector (default : 80)
        num_encoder_layers (int, optional): The number of convolution layers (default : 5)
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        num_channels (int, optional): The number of channels in the convolution filter (default: 256)
        encoder_dim (int, optional): Dimension of encoder output vector (default: 640)
        optimizer (str): Optimizer for training. (default: adam)
    """
    model_name: str = field(default="contextnet", metadata={"help": "Model name"})
    model_size: str = field(default="medium", metadata={"help": "Model size"})
    input_dim: int = field(default=80, metadata={"help": "Dimension of input vector"})
    num_encoder_layers: int = field(default=5, metadata={"help": "The number of convolution layers"})
    kernel_size: int = field(default=5, metadata={"help": "Value of convolution kernel size"})
    num_channels: int = field(default=256, metadata={"help": "The number of channels in the convolution filter"})
    encoder_dim: int = field(default=640, metadata={"help": "Dimension of encoder output vector"})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training"})


@dataclass
class ContextNetLSTMConfigs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.ContextNetLSTM`.

    It is used to initiated an `ContextNetLSTM` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: contextnet_lstm)
        model_size (str, optional): Size of the model['small', 'medium', 'large'] (default : 'medium')
        input_dim (int, optional): Dimension of input vector (default : 80)
        num_encoder_layers (int, optional): The number of convolution layers (default : 5)
        num_decoder_layers (int): The number of decoder layers. (default: 2)
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        num_channels (int, optional): The number of channels in the convolution filter (default: 256)
        encoder_dim (int, optional): Dimension of encoder output vector (default: 640)
        num_attention_heads (int): The number of attention heads. (default: 8)
        attention_dropout_p (float): The dropout probability of attention module. (default: 0.1)
        decoder_dropout_p (float): The dropout probability of decoder. (default: 0.1)
        max_length (int): Max decoding length. (default: 128)
        teacher_forcing_ratio (float): The ratio of teacher forcing. (default: 1.0)
        rnn_type (str): Type of rnn cell (rnn, lstm, gru) (default: lstm)
        decoder_attn_mechanism (str): The attention mechanism for decoder. (default: loc)
        optimizer (str): Optimizer for training. (default: adam)
    """
    model_name: str = field(default="contextnet_lstm", metadata={"help": "Model name"})
    model_size: str = field(default="medium", metadata={"help": "Model size"})
    input_dim: int = field(default=80, metadata={"help": "Dimension of input vector"})
    num_encoder_layers: int = field(default=5, metadata={"help": "The number of convolution layers"})
    num_decoder_layers: int = field(default=2, metadata={"help": "The number of decoder layers."})
    kernel_size: int = field(default=5, metadata={"help": "Value of convolution kernel size"})
    num_channels: int = field(default=256, metadata={"help": "The number of channels in the convolution filter"})
    encoder_dim: int = field(default=640, metadata={"help": "Dimension of encoder output vector"})
    num_attention_heads: int = field(default=8, metadata={"help": "The number of attention heads."})
    attention_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of attention module."})
    decoder_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of decoder."})
    max_length: int = field(default=128, metadata={"help": "Max decoding length."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": "The ratio of teacher forcing. "})
    rnn_type: str = field(default="lstm", metadata={"help": "Type of rnn cell (rnn, lstm, gru)"})
    decoder_attn_mechanism: str = field(default="loc", metadata={"help": "The attention mechanism for decoder."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})


@dataclass
class ContextNetTransducerConfigs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.ContextNetTransducer`.

    It is used to initiated an `ContextNetTransducer` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: contextnet_transducer)
        model_size (str, optional): Size of the model['small', 'medium', 'large'] (default : 'medium')
        input_dim (int, optional): Dimension of input vector (default : 80)
        num_encoder_layers (int, optional): The number of convolution layers (default : 5)
        num_decoder_layers (int, optional): The number of rnn layers (default : 1)
        kernel_size (int, optional): Value of convolution kernel size (default : 5)
        num_channels (int, optional): The number of channels in the convolution filter (default: 256)
        hidden_dim (int, optional): The number of features in the decoder hidden state (default : 2048)
        encoder_dim (int, optional): Dimension of encoder output vector (default: 640)
        decoder_output_dim (int, optional): Dimension of decoder output vector (default: 640)
        dropout (float, optional): Dropout probability of decoder (default: 0.1)
        rnn_type (str, optional): Type of rnn cell (rnn, lstm, gru) (default: lstm)
        optimizer (str): Optimizer for training. (default: adam)
    """
    model_name: str = field(default="contextnet_transducer", metadata={"help": "Model name"})
    model_size: str = field(default="medium", metadata={"help": "Model size"})
    input_dim: int = field(default=80, metadata={"help": "Dimension of input vector"})
    num_encoder_layers: int = field(default=5, metadata={"help": "The number of convolution layers"})
    num_decoder_layers: int = field(default=1, metadata={"help": "The number of rnn layers"})
    kernel_size: int = field(default=5, metadata={"help": "Value of convolution kernel size"})
    num_channels: int = field(default=256, metadata={"help": "The number of channels in the convolution filter"})
    decoder_hidden_state_dim: int = field(
        default=2048, metadata={"help": "The number of features in the decoder hidden state"}
    )
    encoder_dim: int = field(default=640, metadata={"help": "Dimension of encoder output vector"})
    decoder_output_dim: int = field(default=640, metadata={"help": "Dimension of decoder output vector"})
    decoder_dropout_p: float = field(default=0.1, metadata={"help": "Dropout probability of decoder"})
    rnn_type: str = field(default="lstm", metadata={"help": "Type of rnn cell"})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training"})
