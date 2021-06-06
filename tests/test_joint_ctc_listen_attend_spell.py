import unittest
import torch

from openspeech.criterion.joint_ctc_cross_entropy.joint_ctc_cross_entropy import JointCTCCrossEntropyLossConfigs, JointCTCCrossEntropyLoss
from openspeech.models import JointCTCListenAttendSpellModel, JointCTCListenAttendSpellConfigs
from openspeech.utils import build_dummy_configs, DUMMY_INPUTS, DUMMY_TARGETS, DUMMY_INPUT_LENGTHS, DUMMY_TARGET_LENGTHS
from openspeech.vocabs.ksponspeech.character import KsponSpeechCharacterVocabulary


class TestJointCTCListenAttendSpell(unittest.TestCase):
    def test_forward(self):
        configs = build_dummy_configs(
            model_configs=JointCTCListenAttendSpellConfigs(),
            criterion_configs=JointCTCCrossEntropyLossConfigs(),
        )

        vocab = KsponSpeechCharacterVocabulary(configs)
        model = JointCTCListenAttendSpellModel(configs, vocab)
        model.build_model()

        criterion = JointCTCCrossEntropyLoss(configs, num_classes=len(vocab), vocab=vocab)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-04)

        for i in range(3):
            outputs = model(DUMMY_INPUTS, DUMMY_INPUT_LENGTHS)

            loss, ctc_loss, cross_entropy_loss = criterion(
                encoder_logits=outputs["encoder_logits"].transpose(0, 1),
                logits=outputs["logits"],
                output_lengths=outputs["encoder_output_lengths"],
                targets=DUMMY_TARGETS[:, 1:],
                target_lengths=DUMMY_TARGET_LENGTHS,
            )
            loss.backward()
            optimizer.step()
            assert type(loss.item()) == float
            assert type(ctc_loss.item()) == float
            assert type(cross_entropy_loss.item()) == float

    def test_beam_search(self):
        configs = build_dummy_configs(
            model_configs=JointCTCListenAttendSpellConfigs(),
            criterion_configs=JointCTCCrossEntropyLossConfigs(),
        )

        vocab = KsponSpeechCharacterVocabulary(configs)
        model = JointCTCListenAttendSpellModel(configs, vocab)
        model.build_model()
        model.set_beam_decoder(batch_size=3, beam_size=3)

        for i in range(3):
            prediction = model(DUMMY_INPUTS, DUMMY_INPUT_LENGTHS)["predictions"]
            assert isinstance(prediction, torch.Tensor)

    def test_training_step(self):
        configs = build_dummy_configs(
            model_configs=JointCTCListenAttendSpellConfigs(),
            criterion_configs=JointCTCCrossEntropyLossConfigs(),
        )

        vocab = KsponSpeechCharacterVocabulary(configs)
        model = JointCTCListenAttendSpellModel(configs, vocab)
        model.build_model()

        for i in range(3):
            outputs = model.training_step(
                batch=(DUMMY_INPUTS, DUMMY_TARGETS, DUMMY_INPUT_LENGTHS, DUMMY_TARGET_LENGTHS), batch_idx=i
            )
            assert type(outputs["loss"].item()) == float

    def test_validation_step(self):
        configs = build_dummy_configs(
            model_configs=JointCTCListenAttendSpellConfigs(),
            criterion_configs=JointCTCCrossEntropyLossConfigs(),
        )

        vocab = KsponSpeechCharacterVocabulary(configs)
        model = JointCTCListenAttendSpellModel(configs, vocab)
        model.build_model()

        for i in range(3):
            outputs = model.validation_step(
                batch=(DUMMY_INPUTS, DUMMY_TARGETS, DUMMY_INPUT_LENGTHS, DUMMY_TARGET_LENGTHS), batch_idx=i
            )
            assert type(outputs["loss"].item()) == float

    def test_test_step(self):
        configs = build_dummy_configs(
            model_configs=JointCTCListenAttendSpellConfigs(),
            criterion_configs=JointCTCCrossEntropyLossConfigs(),
        )

        vocab = KsponSpeechCharacterVocabulary(configs)
        model = JointCTCListenAttendSpellModel(configs, vocab)
        model.build_model()

        for i in range(3):
            outputs = model.test_step(
                batch=(DUMMY_INPUTS, DUMMY_TARGETS, DUMMY_INPUT_LENGTHS, DUMMY_TARGET_LENGTHS), batch_idx=i
            )
            assert type(outputs["loss"].item()) == float


if __name__ == '__main__':
    unittest.main()
