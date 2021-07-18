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

import os
import hydra
import warnings
import logging
from omegaconf import DictConfig, OmegaConf
from openspeech.metrics import WordErrorRate, CharacterErrorRate
from pytorch_lightning.utilities import rank_zero_info

from openspeech.data.audio.dataset import SpeechToTextDataset
from openspeech.data.sampler import BucketingSampler
from openspeech.data.audio.data_loader import load_dataset, AudioDataLoader
from openspeech.dataclass.initialize import hydra_eval_init
from openspeech.models import MODEL_REGISTRY

logger = logging.getLogger(__name__)


@hydra.main(config_path=os.path.join("..", "openspeech", "configs"), config_name="eval")
def hydra_main(configs: DictConfig) -> None:
    rank_zero_info(OmegaConf.to_yaml(configs))
    wer, cer = 1.0, 1.0

    audio_paths, transcripts = load_dataset(configs.eval.manifest_file_path)

    model = MODEL_REGISTRY[configs.eval.model_name].load_from_checkpoint(configs.eval.checkpoint_path)

    if configs.eval.beam_size > 1:
        model.set_beam_decoder(beam_size=configs.eval.beam_size)

    tokenizer = model.tokenizer

    dataset = SpeechToTextDataset(
        configs=configs,
        dataset_path=configs.eval.dataset_path,
        audio_paths=audio_paths,
        transcripts=transcripts,
        sos_id=tokenizer.sos_id,
        eos_id=tokenizer.eos_id,
    )
    sampler = BucketingSampler(
        data_source=dataset,
        batch_size=configs.eval.batch_size
    )
    data_loader = AudioDataLoader(
        dataset=dataset,
        num_workers=configs.eval.num_workers,
        batch_sampler=sampler,
    )

    wer_metric = WordErrorRate(tokenizer)
    cer_metric = CharacterErrorRate(tokenizer)

    for i, (batch) in enumerate(data_loader):
        inputs, targets, input_lengths, target_lengths = batch

        outputs = model(inputs, input_lengths)

        wer = wer_metric(targets, outputs["predictions"])
        cer = cer_metric(targets, outputs["predictions"])

    logger.info(f"Word Error Rate: {wer}, Character Error Rate: {cer}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    hydra_eval_init()
    hydra_main()
