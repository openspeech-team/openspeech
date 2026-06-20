# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho

"""Tests that the primary Hydra configs dropped the Hydra-1.1-breaking `@package _group_` header."""

from pathlib import Path

import pytest

CONFIG_DIR = Path(__file__).resolve().parents[1] / "openspeech" / "configs"
PRIMARY_CONFIGS = ["train.yaml", "lm_train.yaml", "eval.yaml"]


@pytest.mark.parametrize("config_name", PRIMARY_CONFIGS)
def test_no_package_group_header(config_name):
    text = (CONFIG_DIR / config_name).read_text()
    assert "@package" not in text
    assert "defaults:" in text
