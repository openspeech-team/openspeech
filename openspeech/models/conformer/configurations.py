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
class ConformerConfigs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.Conformer`.

    It is used to initiated an `Conformer` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: conformer)
        encoder_dim (int): Dimension of encoder. (default: 512)
        num_encoder_layers (int): The number of encoder layers. (default: 17)
        num_attention_heads (int): The number of attention heads. (default: 8)
        feed_forward_expansion_factor (int): The expansion factor of feed forward module. (default: 4)
        conv_expansion_factor (int): The expansion factor of convolution module. (default: 2)
        input_dropout_p (float): The dropout probability of inputs. (default: 0.1)
        feed_forward_dropout_p (float): The dropout probability of feed forward module. (default: 0.1)
        attention_dropout_p (float): The dropout probability of attention module. (default: 0.1)
        conv_dropout_p (float): The dropout probability of convolution module. (default: 0.1)
        conv_kernel_size (int): The kernel size of convolution. (default: eq)
        half_step_residual (bool): Flag indication whether to use half step residual or not (default: True)
        optimizer (str): Optimizer for training. (default: adam)
    """
    model_name: str = field(default="conformer", metadata={"help": "Model name"})
    encoder_dim: int = field(default=512, metadata={"help": "Dimension of encoder."})
    num_encoder_layers: int = field(default=17, metadata={"help": "The number of encoder layers."})
    num_attention_heads: int = field(default=8, metadata={"help": "The number of attention heads."})
    feed_forward_expansion_factor: int = field(
        default=4, metadata={"help": "The expansion factor of feed forward module."}
    )
    conv_expansion_factor: int = field(default=2, metadata={"help": "The expansion factor of convolution module."})
    input_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of inputs."})
    feed_forward_dropout_p: float = field(
        default=0.1, metadata={"help": "The dropout probability of feed forward module."}
    )
    attention_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of attention module."})
    conv_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of convolution module."})
    conv_kernel_size: int = field(default=31, metadata={"help": "The kernel size of convolution."})
    half_step_residual: bool = field(
        default=True, metadata={"help": "Flag indication whether to use half step residual or not"}
    )
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})


@dataclass
class ConformerLSTMConfigs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.ConformerLSTM`.

    It is used to initiated an `ConformerLSTM` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: conformer_lstm)
        encoder_dim (int): Dimension of encoder. (default: 512)
        num_encoder_layers (int): The number of encoder layers. (default: 17)
        num_attention_heads (int): The number of attention heads. (default: 8)
        feed_forward_expansion_factor (int): The expansion factor of feed forward module. (default: 4)
        conv_expansion_factor (int): The expansion factor of convolution module. (default: 2)
        input_dropout_p (float): The dropout probability of inputs. (default: 0.1)
        feed_forward_dropout_p (float): The dropout probability of feed forward module. (default: 0.1)
        attention_dropout_p (float): The dropout probability of attention module. (default: 0.1)
        conv_dropout_p (float): The dropout probability of convolution module. (default: 0.1)
        conv_kernel_size (int): The kernel size of convolution. (default: eq)
        half_step_residual (bool): Flag indication whether to use half step residual or not (default: True)
        num_decoder_layers (int): The number of decoder layers. (default: 2)
        decoder_dropout_p (float): The dropout probability of decoder. (default: 0.1)
        max_length (int): Max decoding length. (default: 128)
        teacher_forcing_ratio (float): The ratio of teacher forcing. (default: 1.0)
        rnn_type (str): Type of rnn cell (rnn, lstm, gru) (default: lstm)
        decoder_attn_mechanism (str): The attention mechanism for decoder. (default: loc)
        optimizer (str): Optimizer for training. (default: adam)
    """
    model_name: str = field(default="conformer_lstm", metadata={"help": "Model name"})
    encoder_dim: int = field(default=512, metadata={"help": "Dimension of encoder."})
    num_encoder_layers: int = field(default=17, metadata={"help": "The number of encoder layers."})
    num_attention_heads: int = field(default=8, metadata={"help": "The number of attention heads."})
    feed_forward_expansion_factor: int = field(
        default=4, metadata={"help": "The expansion factor of feed forward module."}
    )
    conv_expansion_factor: int = field(default=2, metadata={"help": "The expansion factor of convolution module."})
    input_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of inputs."})
    feed_forward_dropout_p: float = field(
        default=0.1, metadata={"help": "The dropout probability of feed forward module."}
    )
    attention_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of attention module."})
    conv_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of convolution module."})
    conv_kernel_size: int = field(default=31, metadata={"help": "The kernel size of convolution."})
    half_step_residual: bool = field(
        default=True, metadata={"help": "Flag indication whether to use half step residual or not"}
    )
    num_decoder_layers: int = field(default=2, metadata={"help": "The number of decoder layers."})
    decoder_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of decoder."})
    max_length: int = field(default=128, metadata={"help": "Max decoding length."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": "The ratio of teacher forcing. "})
    rnn_type: str = field(default="lstm", metadata={"help": "Type of rnn cell (rnn, lstm, gru)"})
    decoder_attn_mechanism: str = field(default="loc", metadata={"help": "The attention mechanism for decoder."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})


@dataclass
class ConformerTransducerConfigs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.ConformerTransducer`.

    It is used to initiated an `ConformerTransducer` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: conformer_transducer)
        encoder_dim (int): Dimension of encoder. (default: 512)
        num_encoder_layers (int): The number of encoder layers. (default: 17)
        num_attention_heads (int): The number of attention heads. (default: 8)
        feed_forward_expansion_factor (int): The expansion factor of feed forward module. (default: 4)
        conv_expansion_factor (int): The expansion factor of convolution module. (default: 2)
        input_dropout_p (float): The dropout probability of inputs. (default: 0.1)
        feed_forward_dropout_p (float): The dropout probability of feed forward module. (default: 0.1)
        attention_dropout_p (float): The dropout probability of attention module. (default: 0.1)
        conv_dropout_p (float): The dropout probability of convolution module. (default: 0.1)
        conv_kernel_size (int): The kernel size of convolution. (default: eq)
        half_step_residual (bool): Flag indication whether to use half step residual or not (default: True)
        num_decoder_layers (int): The number of decoder layers. (default: 1)
        decoder_dropout_p (float): The dropout probability of decoder. (default: 0.1)
        max_length (int): Max decoding length. (default: 128)
        teacher_forcing_ratio (float): The ratio of teacher forcing. (default: 1.0)
        rnn_type (str): Type of rnn cell (rnn, lstm, gru) (default: lstm)
        decoder_hidden_state_dim (int): Hidden state dimension of decoder. (default: 640)
        decoder_output_dim (int): Output dimension of decoder. (default: 640)
        optimizer (str): Optimizer for training. (default: adam)
    """
    model_name: str = field(default="conformer_transducer", metadata={"help": "Model name"})
    encoder_dim: int = field(default=512, metadata={"help": "Dimension of encoder."})
    num_encoder_layers: int = field(default=17, metadata={"help": "The number of encoder layers."})
    num_attention_heads: int = field(default=8, metadata={"help": "The number of attention heads."})
    feed_forward_expansion_factor: int = field(
        default=4, metadata={"help": "The expansion factor of feed forward module."}
    )
    conv_expansion_factor: int = field(default=2, metadata={"help": "The expansion factor of convolution module."})
    input_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of inputs."})
    feed_forward_dropout_p: float = field(
        default=0.1, metadata={"help": "The dropout probability of feed forward module."}
    )
    attention_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of attention module."})
    conv_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of convolution module."})
    conv_kernel_size: int = field(default=31, metadata={"help": "The kernel size of convolution."})
    half_step_residual: bool = field(
        default=True, metadata={"help": "Flag indication whether to use half step residual or not"}
    )
    num_decoder_layers: int = field(default=1, metadata={"help": "The number of decoder layers."})
    decoder_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of decoder."})
    max_length: int = field(default=128, metadata={"help": "Max decoding length."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": " The ratio of teacher forcing. "})
    rnn_type: str = field(default="lstm", metadata={"help": "Type of rnn cell (rnn, lstm, gru)"})
    decoder_hidden_state_dim: int = field(default=640, metadata={"help": "Hidden state dimension of decoder."})
    decoder_output_dim: int = field(default=640, metadata={"help": "Output dimension of decoder."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})


@dataclass
class JointCTCConformerLSTMConfigs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.JointCTCConformerLSTM`.

    It is used to initiated an `JointCTCConformerLSTM` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: joint_ctc_conformer_lstm)
        encoder_dim (int): Dimension of encoder. (default: 512)
        num_encoder_layers (int): The number of encoder layers. (default: 17)
        num_attention_heads (int): The number of attention heads. (default: 8)
        feed_forward_expansion_factor (int): The expansion factor of feed forward module. (default: 4)
        conv_expansion_factor (int): The expansion factor of convolution module. (default: 2)
        input_dropout_p (float): The dropout probability of inputs. (default: 0.1)
        feed_forward_dropout_p (float): The dropout probability of feed forward module. (default: 0.1)
        attention_dropout_p (float): The dropout probability of attention module. (default: 0.1)
        conv_dropout_p (float): The dropout probability of convolution module. (default: 0.1)
        conv_kernel_size (int): The kernel size of convolution. (default: eq)
        half_step_residual (bool): Flag indication whether to use half step residual or not (default: True)
        num_decoder_layers (int): The number of decoder layers. (default: 2)
        decoder_dropout_p (float): The dropout probability of decoder. (default: 0.1)
        max_length (int): Max decoding length. (default: 128)
        teacher_forcing_ratio (float): The ratio of teacher forcing. (default: 1.0)
        rnn_type (str): Type of rnn cell (rnn, lstm, gru) (default: lstm)
        decoder_attn_mechanism (str): The attention mechanism for decoder. (default: loc)
        optimizer (str): Optimizer for training. (default: adam)
    """
    model_name: str = field(default="joint_ctc_conformer_lstm", metadata={"help": "Model name"})
    encoder_dim: int = field(default=512, metadata={"help": "Dimension of encoder."})
    num_encoder_layers: int = field(default=17, metadata={"help": "The number of encoder layers."})
    num_attention_heads: int = field(default=8, metadata={"help": "The number of attention heads."})
    feed_forward_expansion_factor: int = field(
        default=4, metadata={"help": "The expansion factor of feed forward module."}
    )
    conv_expansion_factor: int = field(default=2, metadata={"help": "The expansion factor of convolution module."})
    input_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of inputs."})
    feed_forward_dropout_p: float = field(
        default=0.1, metadata={"help": "The dropout probability of feed forward module."}
    )
    attention_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of attention module."})
    conv_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of convolution module."})
    conv_kernel_size: int = field(default=31, metadata={"help": "The kernel size of convolution."})
    half_step_residual: bool = field(
        default=True, metadata={"help": "Flag indication whether to use half step residual or not"}
    )
    num_decoder_layers: int = field(default=2, metadata={"help": "The number of decoder layers."})
    decoder_dropout_p: float = field(default=0.1, metadata={"help": "The dropout probability of decoder."})
    num_decoder_attention_heads: int = field(default=1, metadata={"help": "The number of decoder attention heads."})
    max_length: int = field(default=128, metadata={"help": "Max decoding length."})
    teacher_forcing_ratio: float = field(default=1.0, metadata={"help": " The ratio of teacher forcing. "})
    rnn_type: str = field(default="lstm", metadata={"help": "Type of rnn cell (rnn, lstm, gru)"})
    decoder_attn_mechanism: str = field(default="loc", metadata={"help": "The attention mechanism for decoder."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})
