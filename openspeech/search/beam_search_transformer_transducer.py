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

import torch

from openspeech.decoders import TransformerTransducerDecoder
from openspeech.search.beam_search_base import OpenspeechBeamSearchBase


class BeamSearchTransformerTransducer(OpenspeechBeamSearchBase):
    r"""
    Transformer Transducer Beam Search
    Reference: RNN-T FOR LATENCY CONTROLLED ASR WITH IMPROVED BEAM SEARCH (https://arxiv.org/pdf/1911.01629.pdf)

    Args: joint, decoder, beam_size, expand_beam, state_beam, blank_id
        joint: joint `encoder_outputs` and `decoder_outputs`
        decoder (TransformerTransducerDecoder): base decoder of transformer transducer model.
        beam_size (int): size of beam.
        expand_beam (int): The threshold coefficient to limit the number
        of expanded hypotheses that are added in A (process_hyp).
        state_beam (int): The threshold coefficient in log space to decide if hyps in A (process_hyps)
        is likely to compete with hyps in B (ongoing_beams)
        blank_id (int): blank id

    Inputs: encoder_outputs, max_length
        encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
        max_length (int): max decoding time step

    Returns:
        * predictions (torch.LongTensor): model predictions.
    """

    def __init__(
        self,
        joint,
        decoder: TransformerTransducerDecoder,
        beam_size: int = 3,
        expand_beam: float = 2.3,
        state_beam: float = 4.6,
        blank_id: int = 3,
    ) -> None:
        super(BeamSearchTransformerTransducer, self).__init__(decoder, beam_size)
        self.joint = joint
        self.forward_step = self.decoder.forward_step
        self.expand_beam = expand_beam
        self.state_beam = state_beam
        self.blank_id = blank_id

    def forward(self, encoder_outputs: torch.Tensor, max_length: int):
        r"""
        Beam search decoding.

        Inputs: encoder_outputs, max_length
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
            ``(batch, seq_length, dimension)``
            max_length (int): max decoding time step

        Returns:
            * predictions (torch.LongTensor): model predictions.
        """
        hypothesis = list()
        hypothesis_score = list()

        for batch_idx in range(encoder_outputs.size(0)):
            blank = torch.ones((1, 1), device=encoder_outputs.device, dtype=torch.long) * self.blank_id
            step_input = torch.ones((1, 1), device=encoder_outputs.device, dtype=torch.long) * self.sos_id
            hyp = {
                "prediction": [self.sos_id],
                "logp_score": 0.0,
            }
            ongoing_beams = [hyp]

            for t_step in range(max_length):
                process_hyps = ongoing_beams
                ongoing_beams = list()

                while True:
                    if len(ongoing_beams) >= self.beam_size:
                        break

                    a_best_hyp = max(process_hyps, key=lambda x: x["logp_score"] / len(x["prediction"]))

                    if len(ongoing_beams) > 0:
                        b_best_hyp = max(
                            ongoing_beams,
                            key=lambda x: x["logp_score"] / len(x["prediction"]),
                        )

                        a_best_prob = a_best_hyp["logp_score"]
                        b_best_prob = b_best_hyp["logp_score"]

                        if b_best_prob >= self.state_beam + a_best_prob:
                            break

                    process_hyps.remove(a_best_hyp)

                    step_input[0, 0] = a_best_hyp["prediction"][-1]
                    step_lengths = encoder_outputs.new_tensor([0], dtype=torch.long)

                    step_outputs = self.forward_step(step_input, step_lengths).squeeze(0).squeeze(0)
                    log_probs = self.joint(encoder_outputs[batch_idx, t_step, :], step_outputs)

                    topk_targets, topk_idx = log_probs.topk(k=self.beam_size)

                    if topk_idx[0] != blank:
                        best_logp = topk_targets[0]
                    else:
                        best_logp = topk_targets[1]

                    for j in range(topk_targets.size(0)):
                        topk_hyp = {
                            "prediction": a_best_hyp["prediction"][:],
                            "logp_score": a_best_hyp["logp_score"] + topk_targets[j],
                        }

                        if topk_idx[j] == self.blank_id:
                            ongoing_beams.append(topk_hyp)
                            continue

                        if topk_targets[j] >= best_logp - self.expand_beam:
                            topk_hyp["prediction"].append(topk_idx[j].item())
                            process_hyps.append(topk_hyp)

            ongoing_beams = sorted(
                ongoing_beams,
                key=lambda x: x["logp_score"] / len(x["prediction"]),
                reverse=True,
            )[0]

            hypothesis.append(torch.LongTensor(ongoing_beams["prediction"][1:]))
            hypothesis_score.append(ongoing_beams["logp_score"] / len(ongoing_beams["prediction"]))

        return self._fill_sequence(hypothesis)
