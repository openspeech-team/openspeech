import logging
import unittest

import torch

from openspeech.criterion import Perplexity, PerplexityLossConfigs
from openspeech.lm.lstm_lm import LSTMForLanguageModel
from openspeech.tokenizers.ksponspeech.character import KsponSpeechCharacterTokenizer
from openspeech.utils import DUMMY_TARGETS, build_dummy_configs

logger = logging.getLogger(__name__)


class TestLSTMForLanguageModel(unittest.TestCase):
    def test_lstm_forward(self):
        configs = build_dummy_configs(
            criterion_configs=PerplexityLossConfigs(),
        )
        vocab = KsponSpeechCharacterTokenizer(configs)

        model = LSTMForLanguageModel(
            num_classes=4,
            max_length=32,
            hidden_state_dim=64,
            pad_id=0,
            sos_id=1,
            eos_id=2,
            rnn_type="lstm",
        )

        criterion = Perplexity(configs, vocab)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)

        for i in range(3):
            logits = model(DUMMY_TARGETS, teacher_forcing_ratio=1.0)

            loss = criterion(logits, DUMMY_TARGETS[:, 1:])
            loss.backward()
            optimizer.step()
            assert type(loss.item()) == float

        for i in range(3):
            logits = model(DUMMY_TARGETS, teacher_forcing_ratio=1.0)

            loss = criterion(logits, DUMMY_TARGETS[:, 1:])
            loss.backward()
            optimizer.step()
            assert type(loss.item()) == float

        for i in range(3):
            logits = model(DUMMY_TARGETS, teacher_forcing_ratio=0.0)

            loss = criterion(logits, DUMMY_TARGETS[:, 1:])
            loss.backward()
            optimizer.step()
            assert type(loss.item()) == float

    def test_gru_forward(self):
        configs = build_dummy_configs(
            criterion_configs=PerplexityLossConfigs(),
        )
        vocab = KsponSpeechCharacterTokenizer(configs)

        model = LSTMForLanguageModel(
            num_classes=4,
            max_length=32,
            hidden_state_dim=64,
            pad_id=0,
            sos_id=1,
            eos_id=2,
            rnn_type="gru",
        )

        criterion = Perplexity(configs, vocab)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)

        for i in range(3):
            logits = model(DUMMY_TARGETS, teacher_forcing_ratio=1.0)

            loss = criterion(logits, DUMMY_TARGETS[:, 1:])
            loss.backward()
            optimizer.step()
            assert type(loss.item()) == float

        for i in range(3):
            logits = model(DUMMY_TARGETS, teacher_forcing_ratio=0.0)

            loss = criterion(logits, DUMMY_TARGETS[:, 1:])
            loss.backward()
            optimizer.step()
            assert type(loss.item()) == float


if __name__ == "__main__":
    unittest.main()
