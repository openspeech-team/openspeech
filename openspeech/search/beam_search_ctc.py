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

import torch.nn as nn

from openspeech.utils import CTCDECODE_IMPORT_ERROR


class BeamSearchCTC(nn.Module):
    r"""
    Decodes probability output using ctcdecode package.

    Args:
        labels (list): the tokens you used to train your model
        lm_path (str): the path to your external kenlm language model(LM).
        alpha (int): weighting associated with the LMs probabilities.
        beta (int): weight associated with the number of words within our beam
        cutoff_top_n (int): cutoff number in pruning. Only the top cutoff_top_n characters with the highest probability
            in the vocab will be used in beam search.
        cutoff_prob (float): cutoff probability in pruning. 1.0 means no pruning.
        beam_size (int): this controls how broad the beam search is.
        num_processes (int): parallelize the batch using num_processes workers.
        blank_id (int): this should be the index of the CTC blank token

    Inputs: logits, sizes
        - logits: Tensor of character probabilities, where probs[c,t] is the probability of character c at time t
        - sizes: Size of each sequence in the mini-batch

    Returns:
        - outputs: sequences of the model's best prediction
    """

    def __init__(
        self,
        labels: list,
        lm_path: str = None,
        alpha: int = 0,
        beta: int = 0,
        cutoff_top_n: int = 40,
        cutoff_prob: float = 1.0,
        beam_size: int = 3,
        num_processes: int = 4,
        blank_id: int = 0,
    ) -> None:
        super(BeamSearchCTC, self).__init__()
        try:
            from ctcdecode import CTCBeamDecoder
        except ImportError:
            raise ImportError(CTCDECODE_IMPORT_ERROR)
        assert isinstance(labels, list), "labels must instance of list"
        self.decoder = CTCBeamDecoder(
            labels, lm_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_size, num_processes, blank_id
        )

    def forward(self, logits, sizes=None):
        r"""
        Decodes probability output using ctcdecode package.

        Inputs: logits, sizes
            logits: Tensor of character probabilities, where probs[c,t] is the probability of character c at time t
            sizes: Size of each sequence in the mini-batch

        Returns:
            outputs: sequences of the model's best prediction
        """
        logits = logits.cpu()
        outputs, scores, offsets, seq_lens = self.decoder.decode(logits, sizes)
        return outputs
