import logging
import unittest

import torch

from openspeech.criterion import Perplexity, PerplexityLossConfigs
from openspeech.models.lstm_lm.configurations import LSTMLanguageModelConfigs
from openspeech.models.lstm_lm.model import LSTMLanguageModel
from openspeech.tokenizers.ksponspeech.character import KsponSpeechCharacterTokenizer
from openspeech.utils import DUMMY_LM_INPUTS, DUMMY_LM_TARGETS, build_dummy_configs

logger = logging.getLogger(__name__)


class TestLSTMLanguageModel(unittest.TestCase):
    def test_forward(self):
        configs = build_dummy_configs(
            model_configs=LSTMLanguageModelConfigs,
            criterion_configs=PerplexityLossConfigs(),
        )
        vocab = KsponSpeechCharacterTokenizer(configs)
        model = LSTMLanguageModel(configs, vocab)

        criterion = Perplexity(configs, vocab)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)

        for i in range(3):
            outputs = model(DUMMY_LM_INPUTS)
            loss = criterion(outputs["logits"], DUMMY_LM_TARGETS)
            loss.backward()
            optimizer.step()
            assert type(loss.item()) == float

        for i in range(3):
            outputs = model(DUMMY_LM_INPUTS)
            loss = criterion(outputs["logits"], DUMMY_LM_TARGETS)
            loss.backward()
            optimizer.step()
            assert type(loss.item()) == float

        for i in range(3):
            outputs = model(DUMMY_LM_INPUTS)
            loss = criterion(outputs["logits"], DUMMY_LM_TARGETS)
            loss.backward()
            optimizer.step()
            assert type(loss.item()) == float

    def test_training_step(self):
        configs = build_dummy_configs(
            model_configs=LSTMLanguageModelConfigs,
            criterion_configs=PerplexityLossConfigs(),
        )
        vocab = KsponSpeechCharacterTokenizer(configs)
        model = LSTMLanguageModel(configs, vocab)

        for i in range(5):
            outputs = model.training_step(batch=(DUMMY_LM_INPUTS, DUMMY_LM_TARGETS), batch_idx=i)
            assert type(outputs["perplexity"].item()) == float

    def test_validation_step(self):
        configs = build_dummy_configs(
            model_configs=LSTMLanguageModelConfigs,
            criterion_configs=PerplexityLossConfigs(),
        )
        vocab = KsponSpeechCharacterTokenizer(configs)
        model = LSTMLanguageModel(configs, vocab)

        for i in range(5):
            outputs = model.validation_step(batch=(DUMMY_LM_INPUTS, DUMMY_LM_TARGETS), batch_idx=i)
            assert type(outputs["perplexity"].item()) == float


if __name__ == "__main__":
    unittest.main()
