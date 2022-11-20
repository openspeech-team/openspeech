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

import shutil

import sentencepiece as spm

SENTENCEPIECE_MODEL_PREFIX = "sp"
SENTENCEPIECE_MODEL_TYPE = "bpe"


def train_sentencepiece(transcripts, vocab_size: int = 3200, blank_token: str = "<blank>") -> None:
    print("generate_sentencepiece_input..")

    with open("sentencepiece_input.txt", "w", encoding="utf-8") as f:
        for transcript in transcripts:
            f.write(f"{transcript}\n")

    spm.SentencePieceTrainer.Train(
        f"--input=sentencepiece_input.txt "
        f"--model_prefix={SENTENCEPIECE_MODEL_PREFIX} "
        f"--vocab_size={vocab_size} "
        f"--model_type={SENTENCEPIECE_MODEL_TYPE} "
        f"--pad_id=0 "
        f"--bos_id=1 "
        f"--eos_id=2 "
        f"--unk_id=3 "
        f"--user_defined_symbols={blank_token}"
    )


def convert_subword(transcript: str, sp: spm.SentencePieceProcessor):
    text = " ".join(sp.EncodeAsPieces(transcript))
    label = " ".join([str(sp.PieceToId(token)) for token in text])
    return text, label


def sentence_to_subwords(
    audio_paths: list, transcripts: list, manifest_file_path: str, sp_model_path: str = "sp.model"
) -> None:
    print("sentence_to_subwords...")
    if sp_model_path != f"{SENTENCEPIECE_MODEL_PREFIX}.model":
        shutil.copy(f"{SENTENCEPIECE_MODEL_PREFIX}.model", sp_model_path)

    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)

    with open(manifest_file_path, "w", encoding="utf-8") as f:
        for audio_path, transcript in zip(audio_paths, transcripts):
            audio_path = audio_path.replace("txt", "pcm")
            text, label = convert_subword(transcript, sp)
            f.write(f"{audio_path}\t{text}\t{label}\n")
