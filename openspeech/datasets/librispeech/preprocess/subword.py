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
import shutil

import sentencepiece as spm

from openspeech.datasets.librispeech.preprocess.preprocess import collect_transcripts

SENTENCEPIECE_MODEL_NAME = "sp"


def _prepare_tokenizer(train_transcripts, vocab_size):
    """Prepare sentencepice tokenizer"""
    input_file = "spm_input.txt"
    model_type = "unigram"

    with open(input_file, "w") as f:
        for transcript in train_transcripts:
            f.write(f"{transcript.split('|')[-1]}\n")

    spm.SentencePieceTrainer.Train(
        f"--input={input_file} "
        f"--model_prefix={SENTENCEPIECE_MODEL_NAME} "
        f"--vocab_size={vocab_size} "
        f"--model_type={model_type} "
        f"--pad_id=0 "
        f"--bos_id=1 "
        f"--eos_id=2 "
        f"--unk_id=3 "
        f"--user_defined_symbols=<blank>"
    )


def generate_manifest_files(dataset_path: str, manifest_file_path: str, vocab_path: str, vocab_size: int) -> None:
    """
    Generate manifest files.
    Format: {audio_path}\t{transcript}\t{numerical_label}

    Args:
        vocab_size (int): size of subword vocab

    Returns:
        None
    """
    transcripts_collection = collect_transcripts(dataset_path)
    _prepare_tokenizer(transcripts_collection[0], vocab_size)

    shutil.copy(f"{SENTENCEPIECE_MODEL_NAME}.model", os.path.join(vocab_path, f"{SENTENCEPIECE_MODEL_NAME}.model"))
    shutil.copy(f"{SENTENCEPIECE_MODEL_NAME}.vocab", os.path.join(vocab_path, f"{SENTENCEPIECE_MODEL_NAME}.vocab"))

    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(vocab_path, f"{SENTENCEPIECE_MODEL_NAME}.model"))

    with open(manifest_file_path, "w") as f:
        for idx, part in enumerate(["train-960", "dev-clean", "dev-other", "test-clean", "test-other"]):
            for transcript in transcripts_collection[idx]:
                audio_path, transcript = transcript.split("|")
                text = " ".join(sp.EncodeAsPieces(transcript))
                label = " ".join([str(item) for item in sp.EncodeAsIds(transcript)])
                f.write(f"{audio_path}\t{text}\t{label}\n")
