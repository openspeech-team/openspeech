# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho

"""Regression tests for count_parameters (was `p.numel` instead of `p.numel()`)."""

import pytest
import torch.nn as nn

from openspeech.decoders.openspeech_decoder import OpenspeechDecoder
from openspeech.encoders.openspeech_encoder import OpenspeechEncoder


class _TinyEncoder(OpenspeechEncoder):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)  # 4*3 + 3 = 15 params


class _TinyDecoder(OpenspeechDecoder):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 2)  # 5*2 + 2 = 12 params


@pytest.mark.parametrize("cls,expected", [(_TinyEncoder, 15), (_TinyDecoder, 12)])
def test_count_parameters_returns_int_count(cls, expected):
    module = cls()
    n = module.count_parameters()
    assert isinstance(n, int)
    assert n == sum(p.numel() for p in module.parameters())
    assert n == expected
