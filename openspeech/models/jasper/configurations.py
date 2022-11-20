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
class Jasper5x3Config(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.Jasper5x3`.

    It is used to initiated an `Jasper5x3` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: jasper5x3)
        num_blocks (int): Number of jasper blocks (default: 5)
        num_sub_blocks (int): Number of jasper sub blocks (default: 3)
        in_channels (str): Output channels of jasper block's convolution
        out_channels (str): Output channels of jasper block's convolution
        kernel_size (str): Kernel size of jasper block's convolution
        dilation (str): Dilation of jasper block's convolution
        dropout_p (str): Dropout probability
        optimizer (str): Optimizer for training.
    """
    model_name: str = field(default="jasper5x3", metadata={"help": "Model name"})
    num_blocks: int = field(default=5, metadata={"help": "Number of jasper blocks"})
    num_sub_blocks: int = field(default=3, metadata={"help": "Number of jasper sub blocks"})
    in_channels: str = field(
        default="(None, 256, 256, 256, 384, 384, 512, 512, 640, 640, 768, 768, 896, 1024)",
        metadata={"help": "Input channels of jasper blocks"},
    )
    out_channels: str = field(
        default="(256, 256, 256, 384, 384, 512, 512, 640, 640, 768, 768, 896, 1024, None)",
        metadata={"help": "Output channels of jasper block's convolution"},
    )
    kernel_size: str = field(
        default="(11, 11, 11, 13, 13, 17, 17, 21, 21, 25, 25, 29, 1, 1)",
        metadata={"help": "Kernel size of jasper block's convolution"},
    )
    dilation: str = field(
        default="(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1)",
        metadata={"help": "Dilation of jasper block's convolution"},
    )
    dropout_p: str = field(
        default="(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.0)",
        metadata={"help": "Dropout probability"},
    )
    optimizer: str = field(default="novograd", metadata={"help": "Optimizer for training."})


@dataclass
class Jasper10x5Config(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.Jasper10x5`.

    It is used to initiated an `Jasper10x5` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: jasper10x5)
        num_blocks (int): Number of jasper blocks (default: 10)
        num_sub_blocks (int): Number of jasper sub blocks (default: 5)
        in_channels (str): Output channels of jasper block's convolution
        out_channels (str): Output channels of jasper block's convolution
        kernel_size (str): Kernel size of jasper block's convolution
        dilation (str): Dilation of jasper block's convolution
        dropout_p (str): Dropout probability
        optimizer (str): Optimizer for training.
    """
    model_name: str = field(default="jasper10x5", metadata={"help": "Model name"})
    num_blocks: int = field(default=10, metadata={"help": "Number of jasper blocks"})
    num_sub_blocks: int = field(default=5, metadata={"help": "Number of jasper sub blocks"})
    in_channels: str = field(
        default="(None, 256, 256, 256, 384, 384, 512, 512, 640, 640, 768, 768, 896, 1024)",
        metadata={"help": "Input channels of jasper blocks"},
    )
    out_channels: str = field(
        default="(256, 256, 256, 384, 384, 512, 512, 640, 640, 768, 768, 768, 896, 1024, None)",
        metadata={"help": "Output channels of jasper block's convolution"},
    )
    kernel_size: str = field(
        default="(11, 11, 11, 13, 13, 17, 17, 21, 21, 25, 25, 29, 1, 1)",
        metadata={"help": "Kernel size of jasper block's convolution"},
    )
    dilation: str = field(
        default="(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1)",
        metadata={"help": "Dilation of jasper block's convolution"},
    )
    dropout_p: str = field(
        default="(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 0.3, 0.4, 0.4, 0.0)",
        metadata={"help": "Dropout probability"},
    )
    optimizer: str = field(default="novograd", metadata={"help": "Optimizer for training."})
