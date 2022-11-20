import logging
import unittest

import torch

from openspeech.criterion.label_smoothed_cross_entropy.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyLoss,
    LabelSmoothedCrossEntropyLossConfigs,
)
from openspeech.models import ContextNetLSTMConfigs, ContextNetLSTMModel
from openspeech.tokenizers.ksponspeech.character import KsponSpeechCharacterTokenizer
from openspeech.utils import DUMMY_INPUT_LENGTHS, DUMMY_INPUTS, DUMMY_TARGET_LENGTHS, DUMMY_TARGETS, build_dummy_configs

logger = logging.getLogger(__name__)


class TestContextNetLSTM(unittest.TestCase):
    def test_forward(self):
        configs = build_dummy_configs(
            model_configs=ContextNetLSTMConfigs(),
            criterion_configs=LabelSmoothedCrossEntropyLossConfigs(),
        )

        tokenizer = KsponSpeechCharacterTokenizer(configs)
        model = ContextNetLSTMModel(configs, tokenizer)

        criterion = LabelSmoothedCrossEntropyLoss(configs, num_classes=len(tokenizer), vocab=tokenizer)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)

        for i in range(3):
            outputs = model(DUMMY_INPUTS, DUMMY_INPUT_LENGTHS)

            loss = criterion(outputs["logits"], DUMMY_TARGETS[:, 1:])
            loss.backward()
            optimizer.step()
            print(loss.item())

            assert type(loss.item()) == float

    def test_beam_search(self):
        configs = build_dummy_configs(
            model_configs=ContextNetLSTMConfigs(),
            criterion_configs=LabelSmoothedCrossEntropyLossConfigs(),
        )

        vocab = KsponSpeechCharacterTokenizer(configs)
        model = ContextNetLSTMModel(configs, vocab)
        model.set_beam_decoder(beam_size=3)

        for i in range(3):
            prediction = model(DUMMY_INPUTS, DUMMY_INPUT_LENGTHS)["predictions"]
            assert isinstance(prediction, torch.Tensor)

    def test_training_step(self):
        configs = build_dummy_configs(
            model_configs=ContextNetLSTMConfigs(),
            criterion_configs=LabelSmoothedCrossEntropyLossConfigs(),
        )

        vocab = KsponSpeechCharacterTokenizer(configs)
        model = ContextNetLSTMModel(configs, vocab)

        for i in range(3):
            outputs = model.training_step(
                batch=(DUMMY_INPUTS, DUMMY_TARGETS, DUMMY_INPUT_LENGTHS, DUMMY_TARGET_LENGTHS), batch_idx=i
            )
            assert type(outputs["loss"].item()) == float

    def test_validation_step(self):
        configs = build_dummy_configs(
            model_configs=ContextNetLSTMConfigs(),
            criterion_configs=LabelSmoothedCrossEntropyLossConfigs(),
        )

        vocab = KsponSpeechCharacterTokenizer(configs)
        model = ContextNetLSTMModel(configs, vocab)

        for i in range(3):
            outputs = model.validation_step(
                batch=(DUMMY_INPUTS, DUMMY_TARGETS, DUMMY_INPUT_LENGTHS, DUMMY_TARGET_LENGTHS), batch_idx=i
            )
            assert type(outputs["loss"].item()) == float

    def test_test_step(self):
        configs = build_dummy_configs(
            model_configs=ContextNetLSTMConfigs(),
            criterion_configs=LabelSmoothedCrossEntropyLossConfigs(),
        )

        vocab = KsponSpeechCharacterTokenizer(configs)
        model = ContextNetLSTMModel(configs, vocab)

        for i in range(3):
            outputs = model.test_step(
                batch=(DUMMY_INPUTS, DUMMY_TARGETS, DUMMY_INPUT_LENGTHS, DUMMY_TARGET_LENGTHS), batch_idx=i
            )
            assert type(outputs["loss"].item()) == float


if __name__ == "__main__":
    unittest.main()
