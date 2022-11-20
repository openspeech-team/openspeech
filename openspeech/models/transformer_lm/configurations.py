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
class TransformerLanguageModelConfigs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.TransformerLanguageModel`.

    It is used to initiated an `TransformerLanguageModel` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: transformer_lm)
        num_layers (int): The number of lstm layers. (default: 6)
        d_model (int): The dimension of model. (default: 768)
        dropout_p (float): The dropout probability of encoder. (default: 0.3)
        d_ff (int): Dimenstion of feed forward network. (default: 2048)
        num_attention_heads (int): The number of attention heads. (default: 8)
        max_length (int): Max decoding length. (default: 128)
        optimizer (str): Optimizer for training. (default: adam)
    """
    model_name: str = field(default="transformer_lm", metadata={"help": "Model name"})
    num_layers: int = field(default=6, metadata={"help": "The number of encoder layers."})
    d_model: int = field(default=768, metadata={"help": "The dimension of model."})
    d_ff: int = field(default=1536, metadata={"help": "The dimenstion of feed forward network."})
    num_attention_heads: int = field(default=8, metadata={"help": "The number of attention heads."})
    dropout_p: float = field(default=0.3, metadata={"help": "The dropout probability of encoder."})
    max_length: int = field(default=128, metadata={"help": "Max decoding length."})
    optimizer: str = field(default="adam", metadata={"help": "Optimizer for training."})
