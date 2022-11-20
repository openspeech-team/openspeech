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
class TransformerTransducerConfigs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.TransformerTransducer`.

    It is used to initiated an `TransformerTransducer` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: transformer_transducer)
        extractor (str): The CNN feature extractor. (default: conv2d_subsample)
        d_model (int): Dimension of model. (default: 512)
        d_ff (int): Dimension of feed forward network. (default: 2048)
        num_attention_heads (int): The number of attention heads. (default: 8)
        num_audio_layers (int): The number of audio layers. (default: 18)
        num_label_layers (int): The number of label layers. (default: 2)
        audio_dropout_p (float): The dropout probability of encoder. (default: 0.1)
        label_dropout_p (float): The dropout probability of decoder. (default: 0.1)
        decoder_hidden_state_dim (int): Hidden state dimension of decoder (default: 512)
        decoder_output_dim (int): dimension of model output. (default: 512)
        conv_kernel_size (int): Kernel size of convolution layer. (default: 31)
        max_positional_length (int): Max length of positional encoding. (default: 5000)
        optimizer (str): Optimizer for training. (default: adam)
    """
    model_name: str = field(default="transformer_transducer", metadata={"help": "Model name"})
    encoder_dim: int = field(default=512, metadata={"help": "Dimension of encoder name"})
    d_ff: int = field(default=2048, metadata={"help": "Dimension of feed forward network"})
    num_audio_layers: int = field(default=18, metadata={"help": "Number of audio layers"})
    num_label_layers: int = field(default=2, metadata={"help": "Number of label layers"})
    num_attention_heads: int = field(default=8, metadata={"help": "Number of attention heads"})
    audio_dropout_p: float = field(default=0.1, metadata={"help": "Dropout probability of audio layer"})
    label_dropout_p: float = field(default=0.1, metadata={"help": "Dropout probability of label layer"})
    decoder_hidden_state_dim: int = field(default=512, metadata={"help": "Hidden state dimension of decoder"})
    decoder_output_dim: int = field(default=512, metadata={"help": "Dimension of model output."})
    conv_kernel_size: int = field(default=31, metadata={"help": "Kernel size of convolution layer."})
    max_positional_length: int = field(default=5000, metadata={"help": "Max length of positional encoding."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})
