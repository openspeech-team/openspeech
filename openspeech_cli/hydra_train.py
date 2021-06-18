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
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_info

from openspeech.datasets import DATA_MODULE_REGISTRY
from openspeech.dataclass.initialize import hydra_train_init
from openspeech.models import MODEL_REGISTRY
from openspeech.utils import parse_configs, get_pl_trainer


@hydra.main(config_path=os.path.join("..", "openspeech", "configs"), config_name="train")
def hydra_main(configs: DictConfig) -> None:
    rank_zero_info(OmegaConf.to_yaml(configs))
    pl.seed_everything(configs.trainer.seed)

    logger, num_devices = parse_configs(configs)

    data_module = DATA_MODULE_REGISTRY[configs.dataset.dataset](configs)
    vocab = data_module.prepare_data()
    data_module.setup(vocab=vocab)

    model = MODEL_REGISTRY[configs.model.model_name](configs=configs, vocab=vocab)
    model.build_model()

    trainer = get_pl_trainer(configs, num_devices, logger)
    trainer.fit(model, data_module)
    trainer.test()


if __name__ == '__main__':
    hydra_train_init()
    hydra_main()
