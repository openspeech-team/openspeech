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
    model_name: str = field(
        default="conformer", metadata={"help": "Model name"}
    )
    encoder_dim: int = field(
        default=512, metadata={"help": "Dimension of encoder."}
    )
    num_encoder_layers: int = field(
        default=17, metadata={"help": "The number of encoder layers."}
    )
    num_attention_heads: int = field(
        default=8, metadata={"help": "The number of attention heads."}
    )
    feed_forward_expansion_factor: int = field(
        default=4, metadata={"help": "The expansion factor of feed forward module."}
    )
    conv_expansion_factor: int = field(
        default=2, metadata={"help": "The expansion factor of convolution module."}
    )
    input_dropout_p: float = field(
        default=0.1, metadata={"help": "The dropout probability of inputs."}
    )
    feed_forward_dropout_p: float = field(
        default=0.1, metadata={"help": "The dropout probability of feed forward module."}
    )
    attention_dropout_p: float = field(
        default=0.1, metadata={"help": "The dropout probability of attention module."}
    )
    conv_dropout_p: float = field(
        default=0.1, metadata={"help": "The dropout probability of convolution module."}
    )
    conv_kernel_size: int = field(
        default=31, metadata={"help": "The kernel size of convolution."}
    )
    half_step_residual: bool = field(
        default=True, metadata={"help": "Flag indication whether to use half step residual or not"}
    )
    optimizer: str = field(
        default="adam", metadata={"help": "Optimizer for training."}
    )
