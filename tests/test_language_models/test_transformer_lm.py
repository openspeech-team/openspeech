import logging
import unittest

import torch

from openspeech.criterion import Perplexity, PerplexityLossConfigs
from openspeech.models.transformer_lm.configurations import TransformerLanguageModelConfigs
from openspeech.models.transformer_lm.model import TransformerLanguageModel
from openspeech.tokenizers.ksponspeech.character import KsponSpeechCharacterTokenizer
from openspeech.utils import DUMMY_LM_INPUTS, DUMMY_LM_TARGETS, DYMMY_LM_INPUT_LENGTHS, build_dummy_configs

logger = logging.getLogger(__name__)


class TestTransformerLanguageModel(unittest.TestCase):
    def test_forward(self):
        configs = build_dummy_configs(
            model_configs=TransformerLanguageModelConfigs,
            criterion_configs=PerplexityLossConfigs(),
        )
        vocab = KsponSpeechCharacterTokenizer(configs)
        model = TransformerLanguageModel(configs, vocab)
        model.configure_optimizers()

        criterion = Perplexity(configs, vocab)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)

        for i in range(3):
            outputs = model(DUMMY_LM_INPUTS, DYMMY_LM_INPUT_LENGTHS)
            loss = criterion(outputs["logits"], DUMMY_LM_TARGETS)
            loss.backward()
            optimizer.step()
            assert type(loss.item()) == float

        for i in range(3):
            outputs = model(DUMMY_LM_INPUTS, DYMMY_LM_INPUT_LENGTHS)
            loss = criterion(outputs["logits"], DUMMY_LM_TARGETS)
            loss.backward()
            optimizer.step()
            assert type(loss.item()) == float

        for i in range(3):
            outputs = model(DUMMY_LM_INPUTS, DYMMY_LM_INPUT_LENGTHS)
            loss = criterion(outputs["logits"], DUMMY_LM_TARGETS)
            loss.backward()
            optimizer.step()
            assert type(loss.item()) == float

    def test_training_step(self):
        configs = build_dummy_configs(
            model_configs=TransformerLanguageModelConfigs,
            criterion_configs=PerplexityLossConfigs(),
        )
        vocab = KsponSpeechCharacterTokenizer(configs)
        model = TransformerLanguageModel(configs, vocab)
        model.configure_optimizers()

        for i in range(5):
            outputs = model.training_step(
                batch=(DUMMY_LM_INPUTS, DYMMY_LM_INPUT_LENGTHS, DUMMY_LM_TARGETS), batch_idx=i
            )
            assert type(outputs["perplexity"].item()) == float

    def test_validation_step(self):
        configs = build_dummy_configs(
            model_configs=TransformerLanguageModelConfigs,
            criterion_configs=PerplexityLossConfigs(),
        )
        vocab = KsponSpeechCharacterTokenizer(configs)
        model = TransformerLanguageModel(configs, vocab)
        model.configure_optimizers()

        for i in range(5):
            outputs = model.training_step(
                batch=(DUMMY_LM_INPUTS, DYMMY_LM_INPUT_LENGTHS, DUMMY_LM_TARGETS), batch_idx=i
            )
            assert type(outputs["perplexity"].item()) == float


if __name__ == "__main__":
    unittest.main()
