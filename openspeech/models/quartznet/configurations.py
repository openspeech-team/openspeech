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
class QuartzNet5x5Configs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.QuartzNet5x5`.

    It is used to initiated an `QuartzNet5x5` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: quartznet5x5)
        num_blocks (int): Number of quartznet blocks (default: 5)
        num_sub_blocks (int): Number of quartznet sub blocks (default: 5)
        in_channels (str): Output channels of jasper block's convolution
        out_channels (str): Output channels of jasper block's convolution
        kernel_size (str): Kernel size of jasper block's convolution
        dilation (str): Dilation of jasper block's convolution
        dropout_p (str): Dropout probability
        optimizer (str): Optimizer for training.
    """
    model_name: str = field(default="quartznet5x5", metadata={"help": "Model name"})
    num_blocks: int = field(default=5, metadata={"help": "Number of quartznet blocks"})
    num_sub_blocks: int = field(default=5, metadata={"help": "Number of quartznet sub blocks"})
    in_channels: str = field(
        default="(None, 256, 256, 256, 512, 512, 512, 512, 1024)", metadata={"help": "Input channels of jasper blocks"}
    )
    out_channels: str = field(
        default="(256, 256, 256, 512, 512, 512, 512, 1024, None)",
        metadata={"help": "Output channels of jasper block's convolution"},
    )
    kernel_size: str = field(
        default="(33, 33, 39, 51, 63, 75, 87, 1, 1)", metadata={"help": "Kernel size of jasper block's convolution"}
    )
    dilation: str = field(
        default="(1, 1, 1, 1, 1, 1, 1, 1, 2)", metadata={"help": "Dilation of jasper block's convolution"}
    )
    dropout_p: str = field(
        default="(0.2, None, None, None, None, None, 0.2, 0.2, 0.2)", metadata={"help": "Dropout probability"}
    )
    optimizer: str = field(default="novograd", metadata={"help": "Optimizer for training."})


@dataclass
class QuartzNet10x5Configs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.QuartzNet10x5`.

    It is used to initiated an `QuartzNet10x5` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: quartznet5x5)
        num_blocks (int): Number of quartznet blocks (default: 10)
        num_sub_blocks (int): Number of quartznet sub blocks (default: 5)
        in_channels (str): Output channels of jasper block's convolution
        out_channels (str): Output channels of jasper block's convolution
        kernel_size (str): Kernel size of jasper block's convolution
        dilation (str): Dilation of jasper block's convolution
        dropout_p (str): Dropout probability
        optimizer (str): Optimizer for training.
    """
    model_name: str = field(default="quartznet10x5", metadata={"help": "Model name"})
    num_blocks: int = field(default=10, metadata={"help": "Number of quartznet blocks"})
    num_sub_blocks: int = field(default=5, metadata={"help": "Number of quartznet sub blocks"})
    in_channels: str = field(
        default="(None, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 1024)",
        metadata={"help": "Input channels of jasper blocks"},
    )
    out_channels: str = field(
        default="(256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 1024, None)",
        metadata={"help": "Output channels of jasper block's convolution"},
    )
    kernel_size: str = field(
        default="(33, 33, 33, 39, 39, 51, 51, 63, 63, 75, 75, 87, 1, 1)",
        metadata={"help": "Kernel size of jasper block's convolution"},
    )
    dilation: str = field(
        default="(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2)",
        metadata={"help": "Dilation of jasper block's convolution"},
    )
    dropout_p: str = field(
        default="(0.2, None, None, None, None, None, None, None, None, None, None, 0.2, 0.2, 0.2)",
        metadata={"help": "Dropout probability"},
    )
    optimizer: str = field(default="novograd", metadata={"help": "Optimizer for training."})


@dataclass
class QuartzNet15x5Configs(OpenspeechDataclass):
    r"""
    This is the configuration class to store the configuration of
    a :class:`~openspeech.models.QuartzNet15x5`.

    It is used to initiated an `QuartzNet15x5` model.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Args:
        model_name (str): Model name (default: quartznet15x5)
        num_blocks (int): Number of quartznet blocks (default: 15)
        num_sub_blocks (int): Number of quartznet sub blocks (default: 5)
        in_channels (str): Output channels of jasper block's convolution
        out_channels (str): Output channels of jasper block's convolution
        kernel_size (str): Kernel size of jasper block's convolution
        dilation (str): Dilation of jasper block's convolution
        dropout_p (str): Dropout probability
        optimizer (str): Optimizer for training.
    """
    model_name: str = field(default="quartznet15x5", metadata={"help": "Model name"})
    num_blocks: int = field(default=15, metadata={"help": "Number of quartznet5x5 blocks"})
    num_sub_blocks: int = field(default=5, metadata={"help": "Number of quartznet5x5 sub blocks"})
    in_channels: str = field(
        default="(None, 256, 256, 256, 256, 256, 256, 256, " "512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 1024)",
        metadata={"help": "Input channels of jasper blocks"},
    )
    out_channels: str = field(
        default="(256, 256, 256, 256, 256, 256, 256, " "512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 1024, None)",
        metadata={"help": "Output channels of jasper block's convolution"},
    )
    kernel_size: str = field(
        default="(33, 33, 33, 33, 39, 39, 39, 51, 51, 51, 63, 63, 63, 75, 75, 75, 87, 1, 1)",
        metadata={"help": "Kernel size of jasper block's convolution"},
    )
    dilation: str = field(
        default="(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2)",
        metadata={"help": "Dilation of jasper block's convolution"},
    )
    dropout_p: str = field(
        default="(0.2, None, None, None, None, None, None, None, None, "
        "None, None, None, None, None, None, None, 0.2, 0.2, 0.2)",
        metadata={"help": "Dropout probability"},
    )
    optimizer: str = field(default="novograd", metadata={"help": "Optimizer for training."})
