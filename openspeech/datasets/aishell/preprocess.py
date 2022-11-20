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

import glob
import os
import tarfile

import pandas as pd


def load_label(filepath):
    char2id = dict()
    id2char = dict()

    ch_labels = pd.read_csv(filepath, encoding="utf-8")

    id_list = ch_labels["id"]
    char_list = ch_labels["char"]

    for (id_, char) in zip(id_list, char_list):
        char2id[char] = id_
        id2char[id_] = char
    return char2id, id2char


def read_transcripts(dataset_path):
    """
    Returns:
        transcripts (dict): All the transcripts from AISHELL dataset. They are represented
                            by {audio id: transcript}.
    """
    transcripts_dict = dict()

    with open(os.path.join(dataset_path, "transcript/aishell_transcript_v0.8.txt")) as f:
        for line in f.readlines():
            tokens = line.split()
            audio_path = tokens[0]
            transcript = " ".join(tokens[1:])
            transcripts_dict[audio_path] = transcript

    return transcripts_dict


def sentence_to_target(sentence, char2id):
    target = str()

    for ch in sentence:
        try:
            target += str(char2id[ch]) + " "
        except KeyError:
            continue

    return target[:-1]


def get_key(audio_file):
    """Given an audio file path, return its ID."""
    return os.path.basename(audio_file)[:-4]


def generate_character_labels(dataset_path, vocab_path):
    transcripts, label_list, label_freq = list(), list(), list()

    with open(os.path.join(dataset_path, "transcript/aishell_transcript_v0.8.txt")) as f:
        for line in f.readlines():
            tokens = line.split(" ")
            transcript = " ".join(tokens[1:])
            transcripts.append(transcript)

    for transcript in transcripts:
        for ch in transcript:
            if ch not in label_list:
                label_list.append(ch)
                label_freq.append(1)
            else:
                label_freq[label_list.index(ch)] += 1

    # sort together Using zip
    label_freq, label_list = zip(*sorted(zip(label_freq, label_list), reverse=True))
    label = {"id": [0, 1, 2, 3], "char": ["<pad>", "<sos>", "<eos>", "<blank>"], "freq": [0, 0, 0, 0]}

    for idx, (ch, freq) in enumerate(zip(label_list, label_freq)):
        label["id"].append(idx + 4)
        label["char"].append(ch)
        label["freq"].append(freq)

    label["id"] = label["id"]
    label["char"] = label["char"]
    label["freq"] = label["freq"]

    label_df = pd.DataFrame(label)
    label_df.to_csv(vocab_path, encoding="utf-8", index=False)


def generate_character_script(dataset_path: str, manifest_file_path: str, vocab_path: str):
    tarfiles = glob.glob(os.path.join(dataset_path, f"wav/*.tar.gz"))

    char2id, id2char = load_label(vocab_path)
    transcripts_dict = read_transcripts(dataset_path)

    for f in tarfiles:
        tar = tarfile.open(f, mode="r:gz")
        tar.extractall(os.path.join(dataset_path, "wav"))
        tar.close()
        os.remove(f)

    with open(manifest_file_path, "w") as f:
        for split in ("train", "dev", "test"):
            audio_paths = glob.glob(os.path.join(dataset_path, f"wav/{split}/*/*.wav"))
            keys = [audio_path for audio_path in audio_paths if get_key(audio_path) in transcripts_dict]

            transcripts = [transcripts_dict[get_key(key)] for key in keys]
            labels = [sentence_to_target(transcript, char2id) for transcript in transcripts]

            for idx, audio_path in enumerate(audio_paths):
                audio_paths[idx] = audio_path.replace(f"{dataset_path}/", "")

            # for (audio_path, transcript, label) in zip(audio_paths, transcripts, labels):
            for (audio_path, transcript, label) in zip(keys, transcripts, labels):
                f.write(f"{audio_path}\t{transcript}\t{label}\n")
