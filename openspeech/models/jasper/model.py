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

from omegaconf import DictConfig

from openspeech.encoders import Jasper
from openspeech.models import OpenspeechCTCModel, register_model
from openspeech.models.jasper.configurations import Jasper5x3Config, Jasper10x5Config
from openspeech.tokenizers.tokenizer import Tokenizer


@register_model("jasper5x3", dataclass=Jasper5x3Config)
class Jasper5x3Model(OpenspeechCTCModel):
    r"""
    Jasper: An End-to-End Convolutional Neural Acoustic Model
    Paper: https://arxiv.org/pdf/1904.03288.pdf

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions that contains `y_hats`, `logits`, `output_lengths`
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(Jasper5x3Model, self).__init__(configs, tokenizer)

        self.encoder = Jasper(
            configs=self.configs,
            input_dim=self.configs.audio.num_mels,
            num_classes=self.num_classes,
        )


@register_model("jasper10x5", dataclass=Jasper10x5Config)
class Jasper10x5Model(Jasper5x3Model):
    r"""
    Jasper: An End-to-End Convolutional Neural Acoustic Model
    Paper: https://arxiv.org/pdf/1904.03288.pdf

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokeizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions that contains `y_hats`, `logits`, `output_lengths`
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(Jasper10x5Model, self).__init__(configs, tokenizer)
