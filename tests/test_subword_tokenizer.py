# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho

"""Regression test for the KsponSpeech subword decoder.

Previously the 2-D branch reset `sentence` inside an inner loop, so each row decoded to only its
last token. The fix decodes the whole row at once.
"""

import numpy as np

from openspeech.tokenizers.ksponspeech.subword import KsponSpeechSubwordTokenizer


class _FakeSentencePiece:
    def DecodeIds(self, ids):
        return "|".join(str(i) for i in ids)


def _tokenizer():
    tokenizer = KsponSpeechSubwordTokenizer.__new__(KsponSpeechSubwordTokenizer)
    tokenizer.sp = _FakeSentencePiece()
    return tokenizer


def test_decode_1d_sequence():
    assert _tokenizer().decode(np.array([1, 2, 3])) == "1|2|3"


def test_decode_2d_decodes_each_row_fully():
    out = _tokenizer().decode(np.array([[1, 2], [3, 4]]))
    assert out == ["1|2", "3|4"]
