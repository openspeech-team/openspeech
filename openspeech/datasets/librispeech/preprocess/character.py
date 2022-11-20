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

import pandas as pd

from openspeech.datasets.librispeech.preprocess.preprocess import collect_transcripts

logger = logging.getLogger(__name__)


def _generate_character_labels(labels_dest):
    logger.info("create_char_labels started..")

    special_tokens = ["<pad>", "<sos>", "<eos>", "<blank>"]
    tokens = special_tokens + list(" ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # sort together Using zip
    label = {
        "id": [x for x in range(len(tokens))],
        "char": tokens,
    }

    label_df = pd.DataFrame(label)
    label_df.to_csv(labels_dest, encoding="utf-8", index=False)


def _load_label(filepath):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]

    for (id_, char) in zip(id_list, char_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def sentence_to_target(sentence, char2id):
    target = str()

    for ch in sentence:
        try:
            target += str(char2id[ch]) + " "
        except KeyError:
            continue

    return target[:-1]


def generate_manifest_files(dataset_path: str, manifest_file_path: str, vocab_path: str) -> None:
    """
    Generate manifest files.
    Format: {audio_path}\t{transcript}\t{numerical_label}

    Args:
        vocab_size (int): size of subword vocab

    Returns:
        None
    """
    _generate_character_labels(vocab_path)
    char2id, id2char = _load_label(vocab_path)

    transcripts_collection = collect_transcripts(dataset_path)

    with open(manifest_file_path, "w") as f:
        for idx, part in enumerate(["train-960", "dev-clean", "dev-other", "test-clean", "test-other"]):
            for transcript in transcripts_collection[idx]:
                audio_path, transcript = transcript.split("|")
                label = sentence_to_target(transcript, char2id)
                f.write(f"{audio_path}\t{transcript}\t{label}\n")

    return
