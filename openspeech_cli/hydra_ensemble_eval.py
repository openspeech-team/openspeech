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

import logging
import os
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_info

from openspeech.data.audio.data_loader import AudioDataLoader, load_dataset
from openspeech.data.audio.dataset import SpeechToTextDataset
from openspeech.data.sampler import RandomSampler
from openspeech.dataclass.initialize import hydra_eval_init
from openspeech.metrics import CharacterErrorRate, WordErrorRate
from openspeech.models import MODEL_REGISTRY
from openspeech.search.ensemble_search import EnsembleSearch, WeightedEnsembleSearch

logger = logging.getLogger(__name__)


@hydra.main(config_path=os.path.join("..", "openspeech", "configs"), config_name="eval")
def hydra_main(configs: DictConfig) -> None:
    rank_zero_info(OmegaConf.to_yaml(configs))
    wer, cer = 1.0, 1.0
    models = list()

    audio_paths, transcripts = load_dataset(configs.eval.manifest_file_path)

    model_names = eval(configs.eval.model_names)
    checkpoint_paths = eval(configs.eval.checkpoint_paths)
    ensemble_weights = eval(configs.eval.ensemble_weights)

    for model_name, checkpoint_path in zip(model_names, checkpoint_paths):
        models.append(MODEL_REGISTRY[model_name].load_from_checkpoint(checkpoint_path))

    if configs.eval.beam_size > 1:
        warnings.warn("Currently, Ensemble + beam search is not supports.")

    tokenizer = models[0].tokenizer

    if configs.eval.ensemble_method == "vanilla":
        model = EnsembleSearch(models)
    elif configs.eval.ensemble_method == "weighted":
        model = WeightedEnsembleSearch(models, ensemble_weights)
    else:
        raise ValueError(f"Unsupported ensemble method: {configs.eval.ensemble_method}")

    dataset = SpeechToTextDataset(
        configs=configs,
        dataset_path=configs.eval.dataset_path,
        audio_paths=audio_paths,
        transcripts=transcripts,
        sos_id=tokenizer.sos_id,
        eos_id=tokenizer.eos_id,
    )
    sampler = RandomSampler(data_source=dataset, batch_size=configs.eval.batch_size)
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

        wer = wer_metric(targets, outputs)
        cer = cer_metric(targets, outputs)

    logger.info(f"Word Error Rate: {wer}, Character Error Rate: {cer}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    hydra_eval_init()
    hydra_main()
