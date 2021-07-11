import unittest
import logging

from openspeech.metrics import CharacterErrorRate, WordErrorRate
from openspeech.utils import DUMMY_TARGETS1, DUMMY_TARGETS2, build_dummy_configs
from openspeech.vocabs.ksponspeech.character import KsponSpeechCharacterVocabConfigs, KsponSpeechCharacterVocabulary
from openspeech.vocabs.librispeech.character import LibriSpeechCharacterVocabConfigs, LibriSpeechCharacterVocabulary
from openspeech.vocabs.aishell.character import AIShellCharacterVocabConfigs, AIShellCharacterVocabulary

logger = logging.getLogger(__name__)


class TestMetric(unittest.TestCase):
    def test_kspon_cer(self):
        configs = build_dummy_configs(vocab_configs=KsponSpeechCharacterVocabConfigs())
        vocab = KsponSpeechCharacterVocabulary(configs)

        kspon_cer = CharacterErrorRate(vocab=vocab)
        cer = kspon_cer(targets=DUMMY_TARGETS1, y_hats=DUMMY_TARGETS2)
        print(f"{cer:.3f}")
        assert type(cer) == float

    def test_kspon_wer(self):
        configs = build_dummy_configs(vocab_configs=KsponSpeechCharacterVocabConfigs())
        vocab = KsponSpeechCharacterVocabulary(configs)

        kspon_wer = WordErrorRate(vocab=vocab)
        wer = kspon_wer(targets=DUMMY_TARGETS1, y_hats=DUMMY_TARGETS2)
        print(f"{wer:.3f}")
        assert type(wer) == float

    def test_libri_cer(self):
        configs = build_dummy_configs(vocab_configs=LibriSpeechCharacterVocabConfigs())
        vocab = LibriSpeechCharacterVocabulary(configs)

        libri_cer = CharacterErrorRate(vocab=vocab)
        cer = libri_cer(targets=DUMMY_TARGETS1, y_hats=DUMMY_TARGETS2)
        print(f"{cer:.3f}")
        assert type(cer) == float

    def test_libri_wer(self):
        configs = build_dummy_configs(vocab_configs=LibriSpeechCharacterVocabConfigs())
        vocab = LibriSpeechCharacterVocabulary(configs)

        libri_wer = WordErrorRate(vocab=vocab)
        wer = libri_wer(targets=DUMMY_TARGETS1, y_hats=DUMMY_TARGETS2)
        print(f"{wer:.3f}")
        assert type(wer) == float

    def test_aishell_cer(self):
        configs = build_dummy_configs(vocab_configs=AIShellCharacterVocabConfigs())
        vocab = AIShellCharacterVocabulary(configs)

        aishell_cer = CharacterErrorRate(vocab=vocab)
        cer = aishell_cer(targets=DUMMY_TARGETS1, y_hats=DUMMY_TARGETS2)
        print(f"{cer:.3f}")
        assert type(cer) == float

    def test_aishell_wer(self):
        configs = build_dummy_configs(vocab_configs=AIShellCharacterVocabConfigs())
        vocab = AIShellCharacterVocabulary(configs)

        aishell_wer = WordErrorRate(vocab=vocab)
        wer = aishell_wer(targets=DUMMY_TARGETS1, y_hats=DUMMY_TARGETS2)
        print(f"{wer:.3f}")
        assert type(wer) == float


if __name__ == '__main__':
    unittest.main()