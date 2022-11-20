import logging
import unittest

from openspeech.modules import Conv2dSubsampling
from openspeech.utils import DUMMY_INPUT_LENGTHS, DUMMY_INPUTS

logger = logging.getLogger(__name__)


class TestConformer(unittest.TestCase):
    def test_conv2d_subsampling(self):
        conv_subsample = Conv2dSubsampling(input_dim=80, in_channels=1, out_channels=512)

        outputs, output_lengths = conv_subsample(DUMMY_INPUTS, DUMMY_INPUT_LENGTHS)
        print(f"input lengths : {DUMMY_INPUT_LENGTHS}")
        print(f"output lengths : {output_lengths}")
